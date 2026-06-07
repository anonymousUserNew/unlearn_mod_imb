import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import CremaMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "crema_trained_05.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "crema_unlearned_pleaseee.pth")  # v4
CHECKPOINT_PATH = os.path.join(BASE_DIR, "models", "chk.pth")

BATCH_SIZE = 16
LR         = 1e-5
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 4.0           

C = NUM_CLASSES             

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

# Test loaders for evaluation at the end (if needed)
forget_test_loader = DataLoader(forget_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
retain_test_loader = DataLoader(retain_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
# ──────────────────────────────────────────────────────────────────────────────
# WEIGHT NETWORK  (data-driven alpha predictor)
# ──────────────────────────────────────────────────────────────────────────────
class WeightNet(nn.Module):
    """
    Small MLP that maps a concatenated difference-vector to a scalar score.
    We use the *mean* score over the batch, then pass [a1_score, a2_score, a3_score]
    through softmax * 3 to get normalised weights that sum to 3.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),          # raw scalar score per sample
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)              # (B, 1)

# Difference-vector dimension per branch = 3 vectors * C classes
#   vec = [GT_prob − Student_prob | GT_prob − Teacher_prob | Student_prob − Teacher_prob]
#   shape per branch: (B, 3*C)
#
# a1  → MD  loss   — Fusion branch only   → input_dim = 3*C × 1 branch  = 3C
# a2  → MKR loss   — 3 branches combined  → input_dim = 3*C × 3 branches = 9C
# a3  → UKR loss   — video + audio only   → input_dim = 3*C × 2 branches = 6C

net_a1 = WeightNet(input_dim=9 * C).to(DEVICE)     # MKR fusion
net_a2 = WeightNet(input_dim=6 * C).to(DEVICE)     # UK forget → video + audio

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
model_ori = CremaMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

# Freeze original (teacher)
for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    list(model_unlearn.parameters()) +
    list(net_a1.parameters()) +
    list(net_a2.parameters()),
    lr=LR,
)

# ──────────────────────────────────────────────────────────────────────────────
# LOSS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def masking(logits, forget_labels):
    """Mask the forget class with -inf so the teacher never teaches it."""
    mask = torch.zeros_like(logits)
    mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
    return logits + mask


def Uniform_MD_loss(student_logits, forget_labels, T, reduction="batchmean"):
    """
    Push the fusion branch toward a UNIFORM distribution over the (C-1)
    non-forget classes — exactly 1/(C-1) for each retained class, 0 for
    the forget class.

    This replaces the random-pair teacher approach.  The random-pair teacher
    had its own confident predictions (for whichever class the shuffled
    image+audio pair resembled), causing the student to confidently predict
    that wrong class instead of spreading probability uniformly.
    """
    B, C_dim = student_logits.shape
    # Build uniform target: 1/(C-1) for all classes except the forget class
    target = torch.ones(B, C_dim, device=student_logits.device)
    target[torch.arange(B), forget_labels] = 0.0          # zero forget class
    target = target / target.sum(dim=1, keepdim=True)      # renorm → 1/(C-1) each

    log_prob = F.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_prob, target, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / B

def MD_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    student_log_prob = F.log_softmax(student_logits/T, dim=1)
    masked_teacher_logits = masking(teacher_logits,forget_labels)
    teacher_prob     = F.softmax(masked_teacher_logits / T, dim=1)
    loss = F.kl_div(student_log_prob,teacher_prob, reduction="none")*(T*T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]

def UKR_loss(teacher_logits, student_logits, T, reduction="batchmean"):
    """Uni-modal Knowledge Retention — keep unimodal branches intact on forget set."""
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob     = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]


def MKR_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    """Multi-modal Knowledge Retention — preserve retain-set performance."""
    student_log_prob    = F.log_softmax(student_logits / T, dim=1)
    masked_teacher      = masking(teacher_logits, forget_labels)
    teacher_prob        = F.softmax(masked_teacher / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
# BRANCH SELECTION HELPERS  (same as unlearn.py)
# ──────────────────────────────────────────────────────────────────────────────
# def func_max(out, forget_labels):
#     """
#     Pick the branch with the *highest* confidence on the forget label.
#     Used for MD and UKR (forget data).
#     """
#     all_logits = torch.stack(
#         [out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1
#     )  # (B, 3, C)
#     B = all_logits.size(0)
#     all_probs = F.softmax(all_logits, dim=2)
#     y         = forget_labels.view(B, 1, 1)
#     probs_y   = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
#     best      = probs_y.argmax(dim=1)
#     return all_logits[torch.arange(B, device=all_logits.device), best]

def func_max(out, forget_labels):
    """
    Pick the branch with the *highest* confidence on the forget label,
    comparing only video and audio (not fusion).
    Used for UKR (forget data).
    """
    all_logits = torch.stack(
        [out["audio_logits"], out["video_logits"], out["fusion_logits"]],dim=1
    )  # (B, 2, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y         = forget_labels.view(B, 1, 1)
    probs_y   = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best      = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]


def func_max_retain(out, retain_labels):
    """
    Pick the branch with the *highest* confidence on the retain label.
    Used for MKR (retain data).
    """
    all_logits = torch.stack(
        [out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1
    )  # (B, 3, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y         = retain_labels.view(B, 1, 1)
    probs_y   = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best      = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC ALPHA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_diff_vectors(student_logits, teacher_logits, gt_one_hot):
    """
    Concatenate three difference vectors per sample:
        [GT − Student_prob | GT − Teacher_prob | Student_prob − Teacher_prob]
    Shape: (B, 3*C)
    """
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    diff1 = gt_one_hot  - student_prob
    diff2 = gt_one_hot  - teacher_prob
    diff3 = student_prob - teacher_prob
    return torch.cat([diff1, diff2, diff3], dim=1)   # (B, 3C)


def compute_dynamic_alphas(net_a1, net_a2,
                            out_df_un, out_dr_un,
                            out_df_ori, out_dr_ori,
                            out_ori_random,
                            forget_labels, retain_labels,
                            gt_one_hot_forget, gt_one_hot_retain):
    """
    Compute data-driven alpha weights for [L_MD, L_MKR, L_UKR].

    Strategy (same as unlearn_alpha_old.py):
      1. Build diff vectors for each loss term / branch.
      2. Feed each diff-vector block through the corresponding WeightNet.
      3. Mean-pool over batch to get 3 scalar raw scores.
      4. Softmax * 3 → [a1, a2, a3] (sum = 3, expected 1 each).

    Returns a1, a2, a3 as scalar tensors (differentiable).
    """
    # ── a1: MD loss — Fusion branch, forget data ──────────────────────────────
    # MD target: func_max(out_ori_random, forget_labels)


    # vec_md = get_diff_vectors(
    #     out_df_un["fusion_logits"],
    #     func_max(out_ori_random, forget_labels),
    #     gt_one_hot_forget,
    # )                                               # (B, 3C)
    # raw_a1 = net_a1(vec_md).mean()                 # scalar

    # ── a2: MKR loss — 3 branches, retain data ───────────────────────────────
    # MKR target: func_max_retain(out_dr_ori, retain_labels)
    teacher_mkr = func_max_retain(out_dr_ori, retain_labels)
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], teacher_mkr, gt_one_hot_retain)
    vec_mkr_video  = get_diff_vectors(out_dr_un["video_logits"],  teacher_mkr, gt_one_hot_retain)
    vec_mkr_audio  = get_diff_vectors(out_dr_un["audio_logits"],  teacher_mkr, gt_one_hot_retain)
    vec_mkr = torch.cat([vec_mkr_fusion,vec_mkr_video, vec_mkr_audio], dim=1)  # (B, 9C)
    raw_a1 = net_a1(vec_mkr).mean()                # scalar

    # ── a3: UKR loss — video + audio branches, forget data ───────────────────
    # UKR target: func_max(out_df_ori, forget_labels)
    teacher_ukr = func_max(out_df_ori, forget_labels)
    vec_ukr_video = get_diff_vectors(out_df_un["video_logits"], teacher_ukr, gt_one_hot_forget)
    vec_ukr_audio = get_diff_vectors(out_df_un["audio_logits"], teacher_ukr, gt_one_hot_forget)
    vec_ukr = torch.cat([vec_ukr_video, vec_ukr_audio], dim=1)  # (B, 6C)
    raw_a2 = net_a2(vec_ukr).mean()                # scalar

    # ── Normalise ─────────────────────────────────────────────────────────────
    scores  = torch.stack([raw_a1, raw_a2])   # (3,)
    weights = F.softmax(scores, dim=0) * 3             # sums to 3
    return weights[0], weights[1]

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP:  compute total loss with dynamic alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
                             net_a1, net_a2,
                             batch_df, batch_dr,
                             training: bool = True):
    """
    Compute the three-component unlearning loss with dynamic alpha weights.

    Args:
        training (bool): If True, student forward + WeightNet are in train mode
                         (gradients flow).  If False, wrapped in no_grad for val.

    Returns dict with keys: train_loss, loss_md, loss_multi, loss_uni, a1, a2, a3.
    """
    # Move to device
    for k in batch_df:
        if isinstance(batch_df[k], torch.Tensor):
            batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        if isinstance(batch_dr[k], torch.Tensor):
            batch_dr[k] = batch_dr[k].to(DEVICE)

    forget_labels  = batch_df["label"]
    retain_labels  = batch_dr["label"]

    gt_one_hot_forget = F.one_hot(forget_labels, num_classes=C).float()
    gt_one_hot_retain = F.one_hot(retain_labels, num_classes=C).float()

    # ── Teacher forward (always no_grad) ─────────────────────────────────────
    with torch.no_grad():
        out_df_ori  = model_ori(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori  = model_ori(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

        # Random-pair teacher: shuffle retain audio within batch
        perm       = torch.randperm(batch_df["spectrogram"].size(0))
        rand_spec  = batch_df["spectrogram"][perm]
        out_ori_random = model_ori(batch_df["video"], rand_spec, return_intermediate=True)

    # ── Student forward ───────────────────────────────────────────────────────
    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    # ── Dynamic alphas ────────────────────────────────────────────────────────
    a1, a2 = compute_dynamic_alphas(
        net_a1, net_a2,
        out_df_un, out_dr_un,
        out_df_ori, out_dr_ori,
        out_ori_random,
        forget_labels, retain_labels,
        gt_one_hot_forget, gt_one_hot_retain,
    )

    # ── Individual losses ─────────────────────────────────────────────────────

    # L_MD: push fusion branch toward UNIFORM over the 12 non-forget classes.
    # Previously used a random-pair teacher, which caused the fusion branch to
    # confidently predict whatever class that random pair resembled.  A uniform
    # target directly encodes the desired outcome: max uncertainty on forget data.
    loss_md = Uniform_MD_loss(
        out_df_un["fusion_logits"],
        forget_labels,
        T, reduction="batchmean",
    )

    #loss_md = MD_loss(out_ori_random["fusion_logits"],out_df_un["fusion_logits"],T,forget_labels,reduction="batchmean")

    # L_MKR: all 3 branches on retain data (averaged)
    loss_mkr = (
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["video_logits"],  T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["audio_logits"],  T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["fusion_logits"], T, forget_labels, reduction="batchmean")
    ) / 3.0

    # L_UKR (dynamic alpha): KD from original teacher on forget data.
    # Keeps unimodal distributions close to original model.
    loss_ukr = (
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["video_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["audio_logits"], T, reduction="batchmean")
    )/2.0

    # L_UNI_CE (fixed weight, outside alpha system): direct cross-entropy on
    # L_UNI_CE (fixed weight, outside alpha system): direct cross-entropy on
    # video + audio branches for the forget class.  Ensures unimodal branches
    # always receive a strong gradient to keep classifying correctly
    # regardless of what a3 does.

    # loss_uni_ce =(
    #     F.cross_entropy(out_df_un["video_logits"], forget_labels) +
    #     F.cross_entropy(out_df_un["audio_logits"], forget_labels)
    # )

    # loss_uni_ce = 0.0

    total_loss = (loss_md) + (a1 * loss_mkr) + (a2 * loss_ukr)

    return {
        "train_loss" : total_loss,
        "loss_md"    : loss_md.detach(),
        "loss_multi" : loss_mkr.detach(),
        "loss_uni"   : loss_ukr.detach(),
        "a1"         : a1.item(),
        "a2"         : a2.item(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  with early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None
best_nets_state  = {}

print(f"Starting dynamic-alpha unlearning for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}")
print(f"  Checkpoint → {CHECKPOINT_PATH}\n")

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────────
    model_unlearn.train()
            
    net_a1.train(); net_a2.train()

    running_train_loss = 0.0
    n_train_batches    = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            net_a1, net_a2,
            batch_df, batch_dr,
            training=True,
        )

        out["train_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_unlearn.parameters()) +
            list(net_a1.parameters()) +
            list(net_a2.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches    += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    # ── Validate ───────────────────────────────────────────────────────────────
    model_unlearn.eval()
    net_a1.eval(); net_a2.eval()

    running_val_loss = 0.0
    n_val_batches    = 0
    last_out         = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori, model_unlearn,
                net_a1, net_a2,
                batch_df, batch_dr,
                training=False,
            )
            running_val_loss += out["train_loss"].item()
            n_val_batches    += 1
            last_out          = out

    avg_val_loss = running_val_loss / max(n_val_batches, 1)

    # ── Logging ────────────────────────────────────────────────────────────────
    if last_out is not None:
        print(
            f"Epoch {epoch:3d} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"MD {last_out['loss_md']:.4f} | "
            f"MKR {last_out['loss_multi']:.4f} | "
            f"UKR {last_out['loss_uni']:.4f} | "
            # f"UniCE {last_out['loss_uni_ce']:.4f} | "
            f"a1={last_out['a1']:.3f}  a2={last_out['a2']:.3f}"
        )
    else:
        print(f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    # ── Early Stopping ─────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        best_model_state = deepcopy(model_unlearn.state_dict())
        best_nets_state  = {
            "net_a1": deepcopy(net_a1.state_dict()),
            "net_a2": deepcopy(net_a2.state_dict()),
        }

        # ── Save both artefacts immediately ───────────────────────────────────
        # 1. Just the unlearned model weights (drop-in for eval.py)
        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)

        # 2. Full checkpoint (model + WeightNets + metadata)
        torch.save({
            "epoch"            : epoch,
            "val_loss"         : avg_val_loss,
            "model_state_dict" : model_unlearn.state_dict(),
            "net_a1_state_dict": net_a1.state_dict(),
            "net_a2_state_dict": net_a2.state_dict(),
            "optimizer_state"  : optimizer.state_dict(),
        }, CHECKPOINT_PATH)

        print(f"  --> Best val loss improved to {best_val_loss:.4f}. Models saved.")
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ──────────────────────────────────────────────────────────────────────────────
# RESTORE BEST  &  FINAL SAVE
# ──────────────────────────────────────────────────────────────────────────────
if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    net_a1.load_state_dict(best_nets_state["net_a1"])
    net_a2.load_state_dict(best_nets_state["net_a2"])

    # Overwrite with (definitely) best state
    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    torch.save({
        "epoch"            : "best",
        "val_loss"         : best_val_loss,
        "model_state_dict" : model_unlearn.state_dict(),
        "net_a1_state_dict": net_a1.state_dict(),
        "net_a2_state_dict": net_a2.state_dict(),
    }, CHECKPOINT_PATH)
    print("Restored best model weights.")

print(f"\nUnlearning complete.")
print(f"  Unlearned model : {UNLEARNED_MODEL_PATH}")
print(f"  Full checkpoint : {CHECKPOINT_PATH}")
