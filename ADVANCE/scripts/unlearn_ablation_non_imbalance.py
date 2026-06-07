import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import AdvanceMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "advance_trained_rerun_01.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "advance_unlearned_4loss_01_rerun.pth")
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "models", "advance_chk_ablation.pth")

BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 4.0

C = NUM_CLASSES

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True,  num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True,  num_workers=4)

# Test loaders for evaluation at the end (if needed)
forget_test_loader = DataLoader(forget_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
retain_test_loader = DataLoader(retain_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

# ──────────────────────────────────────────────────────────────────────────────
# WEIGHT NETWORKS  (data-driven alpha predictors)
# ──────────────────────────────────────────────────────────────────────────────
class WeightNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# a1 is fixed to 0 (MD loss weight is constant, not dynamic)
# a2  → MKR  loss  — fusion branch on retain        → input_dim = 3*C
# a3  → UKR  loss  — image + audio on forget        → input_dim = 6*C
# a4  → MKR_UNI    — image + audio on retain        → input_dim = 6*C
net_a1 = WeightNet(input_dim=3 * C).to(DEVICE)   # unused in loss but kept for symmetry
net_a2 = WeightNet(input_dim=3 * C).to(DEVICE)   # MKR fusion
net_a3 = WeightNet(input_dim=6 * C).to(DEVICE)   # UKR unimodal
net_a4 = WeightNet(input_dim=6 * C).to(DEVICE)   # MKR unimodal

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
model_ori = AdvanceMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

# Freeze original (teacher)
for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    list(model_unlearn.parameters()) +
    list(net_a1.parameters()) +
    list(net_a2.parameters()) +
    list(net_a3.parameters()) +
    list(net_a4.parameters()),
    lr=LR,
)

# ──────────────────────────────────────────────────────────────────────────────
# LOSS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def masking(logits, forget_labels):
    """Mask the forget class with -inf so the teacher never teaches it."""
    mask = torch.zeros_like(logits)
    if logits.size(0) == forget_labels.size(0):
        mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
    else:
        for c in forget_labels.unique():
            mask[:, c] = float("-inf")
    return logits + mask


def Uniform_MD_loss(student_logits, forget_labels, T, reduction="batchmean"):
    B, C_dim = student_logits.shape
    target = torch.ones(B, C_dim, device=student_logits.device)
    target[torch.arange(B), forget_labels] = 0.0
    target = target / target.sum(dim=1, keepdim=True)

    log_prob = F.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_prob, target, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / B


def MKR_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    masked_teacher   = masking(teacher_logits, forget_labels)
    teacher_prob     = F.softmax(masked_teacher / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]


def UKR_loss(teacher_logits, student_logits, T, reduction="batchmean"):
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob     = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
# BRANCH SELECTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def func_max(out, forget_labels):
    """Select best branch (audio or image) based on forget class probability."""
    all_logits = torch.stack(
        [out["audio_logits"], out["image_logits"], out["fusion_logits"]], dim=1
    )  # (B, 3, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y = forget_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]


def func_max_retain(out, retain_labels):
    """Select best branch (fusion, audio, or image) based on retain class probability."""
    all_logits = torch.stack(
        [out["fusion_logits"], out["audio_logits"], out["image_logits"]], dim=1
    )  # (B, 3, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y = retain_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC ALPHA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_diff_vectors(student_logits, teacher_logits, gt_one_hot):
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    diff1 = gt_one_hot - student_prob
    diff2 = gt_one_hot - teacher_prob
    diff3 = student_prob - teacher_prob
    return torch.cat([diff1, diff2, diff3], dim=1)   # (B, 3C)


def compute_dynamic_alphas(net_a1, net_a2, net_a3, net_a4,
                            out_df_un, out_dr_un,
                            out_df_ori, out_dr_ori, out_ori_random,
                            forget_labels, retain_labels,
                            gt_one_hot_forget, gt_one_hot_retain):

    # ── a2: MKR loss — fusion branch on retain data ───────────────────────────
    teacher_mkr    = func_max_retain(out_dr_ori, retain_labels)
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], teacher_mkr, gt_one_hot_retain)
    raw_a2         = net_a2(vec_mkr_fusion).mean()

    # ── a4: MKR_UNI — image + audio branches on retain data ──────────────────
    vec_mkr_audio = get_diff_vectors(out_dr_un["audio_logits"], teacher_mkr, gt_one_hot_retain)
    vec_mkr_image = get_diff_vectors(out_dr_un["image_logits"], teacher_mkr, gt_one_hot_retain)
    vec_mkr_uni   = torch.cat([vec_mkr_audio, vec_mkr_image], dim=1)  # (B, 6C)
    raw_a4        = net_a4(vec_mkr_uni).mean()

    # ── a3: UKR loss — image + audio branches on forget data ─────────────────
    teacher_ukr   = func_max(out_df_ori, forget_labels)
    vec_ukr_image = get_diff_vectors(teacher_ukr, out_df_un["image_logits"], gt_one_hot_forget)
    vec_ukr_audio = get_diff_vectors(teacher_ukr, out_df_un["audio_logits"], gt_one_hot_forget)
    vec_ukr       = torch.cat([vec_ukr_image, vec_ukr_audio], dim=1)  # (B, 6C)
    raw_a3        = net_a3(vec_ukr).mean()

    # ── Normalise a2/a3/a4 together (a1 is fixed=0, not dynamic) ─────────────
    scores  = torch.stack([raw_a2, raw_a3, raw_a4])
    weights = F.softmax(scores, dim=0) * 3             # sums to 3
    return weights[0], weights[1], weights[2]          # a2, a3, a4

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP:  compute total loss with dynamic alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
                             net_a1, net_a2, net_a3, net_a4,
                             batch_df, batch_dr,
                             training: bool = True):
    for k in batch_df:
        if isinstance(batch_df[k], torch.Tensor):
            batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        if isinstance(batch_dr[k], torch.Tensor):
            batch_dr[k] = batch_dr[k].to(DEVICE)

    forget_labels = batch_df["label"]
    retain_labels = batch_dr["label"]

    gt_one_hot_forget = F.one_hot(forget_labels, num_classes=C).float()
    gt_one_hot_retain = F.one_hot(retain_labels, num_classes=C).float()

    # ── Teacher forward (always no_grad) ─────────────────────────────────────
    with torch.no_grad():
        out_df_ori = model_ori(batch_df["image"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori = model_ori(batch_dr["image"], batch_dr["spectrogram"], return_intermediate=True)

        perm           = torch.randperm(batch_dr["spectrogram"].size(0))
        rand_spec      = batch_dr["spectrogram"][perm]
        out_ori_random = model_ori(batch_dr["image"], rand_spec, return_intermediate=True)

    # ── Student forward ───────────────────────────────────────────────────────
    out_df_un = model_unlearn(batch_df["image"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["image"], batch_dr["spectrogram"], return_intermediate=True)

    # ── Dynamic alphas (a1 fixed=0, a2/a3/a4 dynamic) ────────────────────────
    a1 = torch.tensor(0.0, device=DEVICE)
    a2, a3, a4 = compute_dynamic_alphas(
        net_a1, net_a2, net_a3, net_a4,
        out_df_un, out_dr_un,
        out_df_ori, out_dr_ori, out_ori_random,
        forget_labels, retain_labels,
        gt_one_hot_forget, gt_one_hot_retain,
    )

    # ── Loss MD: fusion forget → uniform (T=1.0, fixed weight) ───────────────
    loss_md = Uniform_MD_loss(
        out_df_un["fusion_logits"],
        forget_labels,
        1.0, reduction="batchmean",
    )

    # ── Loss MKR: fusion branch on retain (T=2.0, dynamic a2) ────────────────
    teacher_mkr  = func_max_retain(out_dr_ori, retain_labels)
    loss_mkr     = MKR_loss(out_dr_ori["fusion_logits"], out_dr_un["fusion_logits"], 2.0, forget_labels, reduction="batchmean")

    # ── Loss MKR_UNI: image + audio on retain (T=2.0, dynamic a4) ────────────
    loss_mkr_uni = (
        MKR_loss(out_dr_ori["audio_logits"], out_dr_un["audio_logits"], 2.0, forget_labels, reduction="batchmean") +
        MKR_loss(out_dr_ori["image_logits"], out_dr_un["image_logits"], 2.0, forget_labels, reduction="batchmean")
    ) / 2.0

    # ── Loss UKR: image + audio on forget (T=1.0, dynamic a3) ────────────────
    teacher_ukr   = func_max(out_df_ori, forget_labels)
    loss_ukr_image = UKR_loss(out_df_ori, out_df_un["image_logits"], 1.0, reduction="batchmean")
    loss_ukr_audio = UKR_loss(out_df_ori, out_df_un["audio_logits"], 1.0, reduction="batchmean")
    loss_ukr       = (loss_ukr_image + loss_ukr_audio) / 2.0

    # ── CE on unimodal forget branches (fixed, outside alpha system) ──────────
    ukr_loss_ce = (
        F.cross_entropy(out_df_un["image_logits"], forget_labels) +
        F.cross_entropy(out_df_un["audio_logits"], forget_labels)
    )

    total_loss = (loss_md) + (a2 * loss_mkr) + (a3 * loss_ukr) + (a4 * loss_mkr_uni) 

    return {
        "train_loss"   : total_loss,
        "loss_md"      : loss_md.detach(),
        "loss_mkr"     : loss_mkr.detach(),
        "loss_ukr"     : loss_ukr.detach(),
        "loss_mkr_uni" : loss_mkr_uni.detach(),
        "a1"           : a1.item(),
        "a2"           : a2.item(),
        "a3"           : a3.item(),
        "a4"           : a4.item(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  with early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None
best_nets_state  = {}

print(f"Starting dynamic-alpha unlearning (3 losses) for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}")
print(f"  Checkpoint → {CHECKPOINT_PATH}\n")

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────────
    model_unlearn.train()
    net_a1.train(); net_a2.train(); net_a3.train(); net_a4.train()

    running_train_loss = 0.0
    n_train_batches    = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            net_a1, net_a2, net_a3, net_a4,
            batch_df, batch_dr,
            training=True,
        )

        out["train_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_unlearn.parameters()) +
            list(net_a1.parameters()) +
            list(net_a2.parameters()) +
            list(net_a3.parameters()) +
            list(net_a4.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches    += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    # ── Validate ───────────────────────────────────────────────────────────────
    model_unlearn.eval()
    net_a1.eval(); net_a2.eval(); net_a3.eval(); net_a4.eval()

    running_val_loss = 0.0
    n_val_batches    = 0
    last_out         = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori, model_unlearn,
                net_a1, net_a2, net_a3, net_a4,
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
            f"MKR {last_out['loss_mkr']:.4f} | "
            f"UKR {last_out['loss_ukr']:.4f} | "
            f"MKR_UNI {last_out['loss_mkr_uni']:.4f} | "
            f"a1={last_out['a1']:.3f} a2={last_out['a2']:.3f} a3={last_out['a3']:.3f} a4={last_out['a4']:.3f}"
        )
    else:
        print(f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    # ── Early Stopping ─────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0

        best_model_state = deepcopy(model_unlearn.state_dict())
        best_nets_state  = {
            "net_a1": deepcopy(net_a1.state_dict()),
            "net_a2": deepcopy(net_a2.state_dict()),
            "net_a3": deepcopy(net_a3.state_dict()),
            "net_a4": deepcopy(net_a4.state_dict()),
        }

        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        torch.save({
            "epoch"             : epoch,
            "val_loss"          : avg_val_loss,
            "model_state_dict"  : model_unlearn.state_dict(),
            "net_a1_state_dict" : net_a1.state_dict(),
            "net_a2_state_dict" : net_a2.state_dict(),
            "net_a3_state_dict" : net_a3.state_dict(),
            "net_a4_state_dict" : net_a4.state_dict(),
            "optimizer_state"   : optimizer.state_dict(),
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
    net_a3.load_state_dict(best_nets_state["net_a3"])
    net_a4.load_state_dict(best_nets_state["net_a4"])

    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    torch.save({
        "epoch"             : "best",
        "val_loss"          : best_val_loss,
        "model_state_dict"  : model_unlearn.state_dict(),
        "net_a1_state_dict" : net_a1.state_dict(),
        "net_a2_state_dict" : net_a2.state_dict(),
        "net_a3_state_dict" : net_a3.state_dict(),
        "net_a4_state_dict" : net_a4.state_dict(),
    }, CHECKPOINT_PATH)
    print("Restored best model weights.")

print(f"\nUnlearning complete.")
print(f"  Unlearned model : {UNLEARNED_MODEL_PATH}")
print(f"  Full checkpoint : {CHECKPOINT_PATH}")