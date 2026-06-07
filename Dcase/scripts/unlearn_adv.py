import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

# class GradientReversal(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg() * ctx.alpha, None

# def revgrad(x, alpha=1.0):
#     return GradientReversal.apply(x, alpha)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import DcaseMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "dcase_unlearned_warmup.pth")
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "models", "dcase_chk_ablation.pth")

BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 4.0           

# ── WARMUP CONFIGURATION ───────────────────────────────────────────────────────
WARMUP_EPOCHS = 5             # Epochs to optimize alphas only (freeze model)
ALPHA_LR = 1e-2               # Learning rate for alpha optimization (higher than model LR)

C = NUM_CLASSES             

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

# ──────────────────────────────────────────────────────────────────────────────
# WEIGHT NETWORK  (data-driven alpha predictor)
# ──────────────────────────────────────────────────────────────────────────────
class WeightNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),          # raw scalar score per sample
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)              # (B, 1)

net_a1 = WeightNet(input_dim=3 * C).to(DEVICE)     # MD  → fusion on forget
net_a2 = WeightNet(input_dim=3 * C).to(DEVICE)     # MKR → fusion on retain (smaller now - fusion only)
net_a3 = WeightNet(input_dim=6 * C).to(DEVICE)     # UKR → video + audio on forget
net_a4 = WeightNet(input_dim=6 * C).to(DEVICE)     # UKR → video + audio on retain (CROSS-MODAL LEARNING)

# ── LEARNABLE ALPHA COEFFICIENTS (for warmup phase) ──────────────────────────
# Start with balanced initialization
alpha_md = nn.Parameter(torch.tensor(2.0, device=DEVICE, dtype=torch.float32))
alpha_mkr = nn.Parameter(torch.tensor(1.0, device=DEVICE, dtype=torch.float32))
alpha_ukr_forget = nn.Parameter(torch.tensor(1.5, device=DEVICE, dtype=torch.float32))
alpha_ukr_retain = nn.Parameter(torch.tensor(1.5, device=DEVICE, dtype=torch.float32))

# Store learned alphas to use in main phase
learned_alphas = {
    "a1": alpha_md.data.clone(),
    "a2": alpha_mkr.data.clone(),
    "a3": alpha_ukr_forget.data.clone(),
    "a4": alpha_ukr_retain.data.clone(),
}

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
model_ori = DcaseMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

# Freeze original (teacher)
for p in model_ori.parameters():
    p.requires_grad = False

# Freeze model during warmup
for p in model_unlearn.parameters():
    p.requires_grad = True

# ── WARMUP OPTIMIZER: Only optimize alpha coefficients ───────────────────────
optimizer_warmup = torch.optim.Adam([alpha_md, alpha_mkr, alpha_ukr_forget, alpha_ukr_retain], lr=ALPHA_LR)

# ── MAIN OPTIMIZER: Optimize model + networks + alphas ──────────────────────
optimizer_main = torch.optim.Adam(
    list(model_unlearn.parameters()) +
    list(net_a1.parameters()) +
    list(net_a2.parameters()) +
    list(net_a3.parameters()) +
    list(net_a4.parameters()) +
    [alpha_md, alpha_mkr, alpha_ukr_forget, alpha_ukr_retain],
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
    # Build uniform target: 1/(C-1) for all classes except the forget class
    target = torch.ones(B, C_dim, device=student_logits.device)
    target[torch.arange(B), forget_labels] = 0.0          # zero forget class
    target = target / target.sum(dim=1, keepdim=True)     # renorm → 1/(C-1) each

    log_prob = F.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_prob, target, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / B

def MD_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    student_log_prob = F.log_softmax(student_logits/T, dim=1)
    #masked_teacher_logits = masking(teacher_logits,forget_labels)
    teacher_prob     = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(student_log_prob,teacher_prob, reduction="none")*(T*T)
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


def MKR_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    student_log_prob    = F.log_softmax(student_logits / T, dim=1)
    masked_teacher      = masking(teacher_logits, forget_labels)
    teacher_prob        = F.softmax(teacher_logits / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
# BRANCH SELECTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def func_max(out, forget_labels):
    all_logits = torch.stack(
        [out["audio_logits"], out["video_logits"]], dim=1
    )  # (B, 2, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y         = forget_labels.view(B, 1, 1)
    probs_y   = all_probs.gather(dim=2, index=y.expand(B, 2, 1)).squeeze(2)
    best      = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]


def func_max_retain(out, retain_labels):
    all_logits = torch.stack(
        [out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1
    )  # (B, 3, C)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y         = retain_labels.view(B, 1, 1)
    probs_y   = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best      = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]

# NEW: For cross-modal learning - audio learns from video specifically
def func_video_only(out, labels=None):
    """Return video logits (teacher for weak audio student)"""
    return out["video_logits"]

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC ALPHA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_diff_vectors(student_logits, teacher_logits, gt_one_hot):
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    diff1 = gt_one_hot  - student_prob
    diff2 = gt_one_hot  - teacher_prob
    diff3 = student_prob - teacher_prob
    return torch.cat([diff1, diff2, diff3], dim=1)   # (B, 3C)


def compute_dynamic_alphas(net_a1, net_a2, net_a3, net_a4,
                            out_df_un, out_dr_un,
                            out_df_ori, out_dr_ori,
                            forget_labels, retain_labels,
                            gt_one_hot_forget, gt_one_hot_retain):
    # ── a1: MD loss — Fusion branch, forget data ──────────────────────────────
    target_logits = torch.zeros_like(out_df_un["fusion_logits"])
    target_logits[torch.arange(target_logits.size(0)), forget_labels] = float("-inf")

    vec_md = get_diff_vectors(
        out_df_un["fusion_logits"],
        target_logits,
        gt_one_hot_forget,
    )                                               # (B, 3C)
    raw_a1 = net_a1(vec_md).mean()         # maximize weight for tough loss

    # ── a2: MKR loss — Fusion only, retain data (protect against re-learning) ─
    teacher_mkr = func_max_retain(out_dr_ori, retain_labels)
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], teacher_mkr, gt_one_hot_retain)
    raw_a2 = net_a2(vec_mkr_fusion).mean()  # scalar

    # ── a3: UKR loss — video + audio branches, forget data ───────────────────
    teacher_ukr = func_max(out_df_ori, forget_labels)
    vec_ukr_video = get_diff_vectors(out_df_un["video_logits"], teacher_ukr, gt_one_hot_forget)
    vec_ukr_audio = get_diff_vectors(out_df_un["audio_logits"], teacher_ukr, gt_one_hot_forget)
    vec_ukr = torch.cat([vec_ukr_video, vec_ukr_audio], dim=1)  # (B, 6C)
    raw_a3 = net_a3(vec_ukr).mean()        # scalar

    # ── a4: UKR_retain loss — cross-modal learning on retain ───────────────────
    # CRITICAL: This is dedicated to audio learning from video on retain set
    teacher_video = func_video_only(out_dr_ori)
    vec_ukr_retain_video = get_diff_vectors(out_dr_un["video_logits"], teacher_video, gt_one_hot_retain)
    vec_ukr_retain_audio = get_diff_vectors(out_dr_un["audio_logits"], teacher_video, gt_one_hot_retain)
    vec_ukr_retain = torch.cat([vec_ukr_retain_video, vec_ukr_retain_audio], dim=1)  # (B, 6C)
    raw_a4 = net_a4(vec_ukr_retain).mean()  # scalar

    # ── Normalise ─────────────────────────────────────────────────────────────
    scores  = torch.stack([raw_a1, raw_a2, raw_a3, raw_a4])    # (4,)
    weights = F.softmax(scores, dim=0) * 4             # sums to 4
    weights[3] = weights[3] * 2.0                          # boost a4 (cross-modal retain) since it's most critical for audio
    return weights[0], weights[1], weights[2], weights[3]

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP:  compute total loss with dynamic alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
                             net_a1, net_a2, net_a3, net_a4,
                             batch_df, batch_dr,
                             training: bool = True,
                             use_learnable_alphas: bool = True,
                             fixed_alphas: dict = None):
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

    with torch.no_grad():
        out_df_ori  = model_ori(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori  = model_ori(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

        perm       = torch.randperm(batch_df["spectrogram"].size(0))
        rand_spec  = batch_df["spectrogram"][perm]
        out_ori_random = model_ori(batch_df["video"], rand_spec, return_intermediate=True)

    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    # Use either learnable alphas (warmup) or fixed alphas (main phase)
    if use_learnable_alphas:
        # Use the learnable alpha parameters directly
        a1 = torch.clamp(alpha_md, min=0.1, max=10.0)      # Prevent extreme values
        a2 = torch.clamp(alpha_mkr, min=0.1, max=10.0)
        a3 = torch.clamp(alpha_ukr_forget, min=0.1, max=10.0)
        a4 = torch.clamp(alpha_ukr_retain, min=0.1, max=10.0)
    else:
        # Use fixed alphas from warmup phase
        a1 = torch.tensor(fixed_alphas["a1"], device=DEVICE, dtype=torch.float32)
        a2 = torch.tensor(fixed_alphas["a2"], device=DEVICE, dtype=torch.float32)
        a3 = torch.tensor(fixed_alphas["a3"], device=DEVICE, dtype=torch.float32)
        a4 = torch.tensor(fixed_alphas["a4"], device=DEVICE, dtype=torch.float32)
        # Still compute networks for additional features, but alphas are frozen
        a1_dyn, a2_dyn, a3_dyn, a4_dyn = compute_dynamic_alphas(
            net_a1, net_a2, net_a3, net_a4,
            out_df_un, out_dr_un,
            out_df_ori, out_dr_ori,
            forget_labels, retain_labels,
            gt_one_hot_forget, gt_one_hot_retain,
        )

    loss_md = Uniform_MD_loss(
        out_df_un["fusion_logits"],
        forget_labels,
        T, reduction="batchmean",
    )

    # ── Loss terms (each with dedicated alpha) ──────────────────────────────────
    
    # a1 * loss_md: Forget fusion with uniform distribution
    
    # a2 * loss_mkr_fusion: Protect fusion on retain set (don't re-learn forget)
    loss_mkr_fusion = MKR_loss(
        func_max_retain(out_dr_ori, retain_labels), 
        out_dr_un["fusion_logits"], 
        T, forget_labels, reduction="batchmean"
    )
    
    # a3 * loss_ukr_forget: Cross-modal on forget (audio learns from video)
    loss_ukr_forget = 0.5 * (
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["video_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["audio_logits"], T, reduction="batchmean")
    )
    
    # a4 * loss_ukr_retain: CRITICAL - Cross-modal on retain (audio learns from video)
    # This is the KEY to improving audio on retain set
    loss_ukr_retain = (
        UKR_loss(func_video_only(out_dr_ori), out_dr_un["video_logits"], T, reduction="batchmean") +
        UKR_loss(func_video_only(out_dr_ori), out_dr_un["audio_logits"], T, reduction="batchmean")
    )/2.0
    
    total_loss = (a1*loss_md) + (a2*loss_mkr_fusion) + (a3*loss_ukr_forget) + (a4*loss_ukr_retain)

    return {
        "train_loss" : total_loss,
        "loss_md"    : loss_md.detach(),
        "loss_mkr"   : loss_mkr_fusion.detach(),
        "loss_ukr_forget" : loss_ukr_forget.detach(),
        "loss_ukr_retain" : loss_ukr_retain.detach(),
        "a1"         : a1.item(),
        "a2"         : a2.item(),
        "a3"         : a3.item(),
        "a4"         : a4.item(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  with early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None
best_nets_state  = {}

print(f"Starting alpha-warmup unlearning for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  Warmup phase: {WARMUP_EPOCHS} epochs (optimize alphas only, frozen model)")
print(f"  Main phase: {EPOCHS - WARMUP_EPOCHS} epochs (optimize model with fixed alphas)")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}")
print(f"  Checkpoint → {CHECKPOINT_PATH}\n")

for epoch in range(EPOCHS):
    is_warmup = epoch < WARMUP_EPOCHS
    
    if is_warmup:
        # WARMUP PHASE: Freeze model, only optimize alphas
        model_unlearn.eval()
        for p in model_unlearn.parameters():
            p.requires_grad = False
        for net in [net_a1, net_a2, net_a3, net_a4]:
            net.train()
        optimizer = optimizer_warmup
        print(f"\n[WARMUP {epoch + 1}/{WARMUP_EPOCHS}] Optimizing alpha coefficients...")
    else:
        # MAIN PHASE: Unfreeze model, use fixed alphas from warmup
        model_unlearn.train()
        for p in model_unlearn.parameters():
            p.requires_grad = True
        for net in [net_a1, net_a2, net_a3, net_a4]:
            net.train()
        optimizer = optimizer_main
        
        # Save learned alphas for main phase
        if epoch == WARMUP_EPOCHS:
            learned_alphas["a1"] = alpha_md.data.clone().item()
            learned_alphas["a2"] = alpha_mkr.data.clone().item()
            learned_alphas["a3"] = alpha_ukr_forget.data.clone().item()
            learned_alphas["a4"] = alpha_ukr_retain.data.clone().item()
            print(f"\n[MAIN PHASE] Starting with learned alphas:")
            print(f"  a1 (MD): {learned_alphas['a1']:.4f}")
            print(f"  a2 (MKR): {learned_alphas['a2']:.4f}")
            print(f"  a3 (UKR_forget): {learned_alphas['a3']:.4f}")
            print(f"  a4 (UKR_retain): {learned_alphas['a4']:.4f}\n")

    running_train_loss = 0.0
    n_train_batches    = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            net_a1, net_a2, net_a3, net_a4,
            batch_df, batch_dr,
            training=True,
            use_learnable_alphas=is_warmup,
            fixed_alphas=None if is_warmup else learned_alphas,
        )

        out["train_loss"].backward()
        
        if is_warmup:
            # Only clip alpha gradients in warmup
            torch.nn.utils.clip_grad_norm_([alpha_md, alpha_mkr, alpha_ukr_forget, alpha_ukr_retain], max_norm=1.0)
        else:
            # Clip all gradients in main phase
            torch.nn.utils.clip_grad_norm_(
                list(model_unlearn.parameters()) +
                list(net_a1.parameters()) +
                list(net_a2.parameters()) +
                list(net_a3.parameters()) +
                list(net_a4.parameters()) +
                [alpha_md, alpha_mkr, alpha_ukr_forget, alpha_ukr_retain],
                max_norm=1.0,
            )
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches    += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

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
                use_learnable_alphas=is_warmup,
                fixed_alphas=None if is_warmup else learned_alphas,
            )
            running_val_loss += out["train_loss"].item()
            n_val_batches    += 1
            last_out          = out

    avg_val_loss = running_val_loss / max(n_val_batches, 1)

    if last_out is not None:
        print(
            f"Epoch {epoch:3d} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"MD {last_out['loss_md']:.4f} | "
            f"MKR {last_out['loss_mkr']:.4f} | "
            f"UKR_f {last_out['loss_ukr_forget']:.4f} | "
            f"UKR_r {last_out['loss_ukr_retain']:.4f} | "
            f"a1={last_out['a1']:.3f} a2={last_out['a2']:.3f} a3={last_out['a3']:.3f} a4={last_out['a4']:.3f}"
        )
    else:
        print(f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
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
            "epoch"            : epoch,
            "val_loss"         : avg_val_loss,
            "model_state_dict" : model_unlearn.state_dict(),
            "net_a1_state_dict": net_a1.state_dict(),
            "net_a2_state_dict": net_a2.state_dict(),
            "net_a3_state_dict": net_a3.state_dict(),
            "net_a4_state_dict": net_a4.state_dict(),
            "optimizer_state"  : optimizer.state_dict(),
        }, CHECKPOINT_PATH)

        print(f"  --> Best val loss improved to {best_val_loss:.4f}. Models saved.")
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    net_a1.load_state_dict(best_nets_state["net_a1"])
    net_a2.load_state_dict(best_nets_state["net_a2"])
    net_a3.load_state_dict(best_nets_state["net_a3"])
    net_a4.load_state_dict(best_nets_state["net_a4"])

    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    torch.save({
        "epoch"            : "best",
        "val_loss"         : best_val_loss,
        "model_state_dict" : model_unlearn.state_dict(),
        "net_a1_state_dict": net_a1.state_dict(),
        "net_a2_state_dict": net_a2.state_dict(),
        "net_a3_state_dict": net_a3.state_dict(),
        "net_a4_state_dict": net_a4.state_dict(),
    }, CHECKPOINT_PATH)
    print("Restored best model weights.")

print(f"\nUnlearning complete.")
print(f"  Unlearned model : {UNLEARNED_MODEL_PATH}")
print(f"  Full checkpoint : {CHECKPOINT_PATH}")
