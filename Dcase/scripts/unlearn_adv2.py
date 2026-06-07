import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import DcaseMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "dcase_unlearned_vBranch.pth")
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "models", "dcase_chk_adv2.pth")

BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 4.0           

C = NUM_CLASSES             

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

# ──────────────────────────────────────────────────────────────────────────────
# DATA SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"DATA LOADING SUMMARY")
print(f"{'='*80}")
print(f"Forget - Train: {len(forget_train)} samples | Val: {len(forget_val)} samples")
print(f"Retain - Train: {len(retain_train)} samples | Val: {len(retain_val)} samples")
print(f"\nWith BATCH_SIZE={BATCH_SIZE} and drop_last=False:")
print(f"  Forget Train Batches: {(len(forget_train) + BATCH_SIZE - 1) // BATCH_SIZE}")
print(f"  Forget Val Batches:   {(len(forget_val) + BATCH_SIZE - 1) // BATCH_SIZE}")
print(f"  Retain Train Batches: {(len(retain_train) + BATCH_SIZE - 1) // BATCH_SIZE}")
print(f"  Retain Val Batches:   {(len(retain_val) + BATCH_SIZE - 1) // BATCH_SIZE}")
print(f"{'='*80}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 6 SEPARATE WEIGHT NETWORKS (one alpha per component)
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

net_a1 = WeightNet(input_dim=3 * C).to(DEVICE)     # a1: MD loss (forget fusion)
net_a2 = WeightNet(input_dim=3 * C).to(DEVICE)     # a2: MKR loss (retain fusion protection)
net_a3 = WeightNet(input_dim=3 * C).to(DEVICE)     # a3: UKR video forget
net_a4 = WeightNet(input_dim=3 * C).to(DEVICE)     # a4: UKR audio forget
net_a5 = WeightNet(input_dim=3 * C).to(DEVICE)     # a5: UKR video retain
net_a6 = WeightNet(input_dim=3 * C).to(DEVICE)     # a6: UKR audio retain

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
model_ori = DcaseMultimodalModel(num_classes=C).to(DEVICE)
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
    list(net_a4.parameters()) +
    list(net_a5.parameters()) +
    list(net_a6.parameters()),
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
    target[torch.arange(B), forget_labels] = 0.0
    target = target / target.sum(dim=1, keepdim=True)

    log_prob = F.log_softmax(student_logits / T, dim=1)
    loss = F.kl_div(log_prob, target, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / B

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
    teacher_prob        = F.softmax(masked_teacher / T, dim=1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    if reduction == "none":
        return loss.sum(dim=1)
    return loss.sum() / student_logits.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
# BRANCH SELECTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def func_max(out, forget_labels):
    all_logits = torch.stack([out["audio_logits"], out["video_logits"]], dim=1)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y = forget_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 2, 1)).squeeze(2)
    best = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]

def func_max_retain(out, retain_labels):
    all_logits = torch.stack([out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1)
    B = all_logits.size(0)
    all_probs = F.softmax(all_logits, dim=2)
    y = retain_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best = probs_y.argmax(dim=1)
    return all_logits[torch.arange(B, device=all_logits.device), best]

def func_video_only(out, labels=None):
    """Return video logits (teacher for weak audio student)"""
    return out["video_logits"]

# def func_video_only(out, retain_labels):
#     all_logits = torch.stack([out["audio_logits"], out["video_logits"]], dim=1)
#     B = all_logits.size(0)
#     all_probs = F.softmax(all_logits, dim=2)
#     y = retain_labels.view(B, 1, 1)
#     probs_y = all_probs.gather(dim=2, index=y.expand(B, 2, 1)).squeeze(2)
#     best = probs_y.argmax(dim=1)
#     return all_logits[torch.arange(B, device=all_logits.device), best]

# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC ALPHA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_diff_vectors(student_logits, teacher_logits, gt_one_hot):
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    diff1 = gt_one_hot - student_prob
    diff2 = gt_one_hot - teacher_prob
    diff3 = student_prob - teacher_prob
    return torch.cat([diff1, diff2, diff3], dim=1)

def compute_dynamic_alphas(net_a1, net_a2, net_a3, net_a4, net_a5, net_a6,
                            out_df_un, out_dr_un,
                            out_df_ori, out_dr_ori,
                            forget_labels, retain_labels,
                            gt_one_hot_forget, gt_one_hot_retain):
    # ── a1: MD loss — Forget fusion ────────────────────────────────────────────
    target_logits = torch.zeros_like(out_df_un["fusion_logits"])
    target_logits[torch.arange(target_logits.size(0)), forget_labels] = float("-inf")
    vec_md = get_diff_vectors(out_df_un["fusion_logits"], target_logits, gt_one_hot_forget)
    raw_a1 = net_a1(vec_md).mean()

    # ── a2: MKR loss — Retain fusion protection ────────────────────────────────
    teacher_mkr = func_max_retain(out_dr_ori, retain_labels)
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], teacher_mkr, gt_one_hot_retain)
    raw_a2 = net_a2(vec_mkr_fusion).mean()

    # ── a3: UKR video forget (learn from stable video teacher) ──────────────────────────────
    teacher_video_forget = func_max(out_df_ori, forget_labels)
    vec_video_forget = get_diff_vectors(out_df_un["video_logits"], teacher_video_forget, gt_one_hot_forget)
    raw_a3 = net_a3(vec_video_forget).mean()

    # ── a4: UKR audio forget (learn from strong video teacher) ──────────────────────────────
    teacher_audio_forget = func_max(out_df_ori, forget_labels)
    vec_audio_forget = get_diff_vectors(out_df_un["audio_logits"], teacher_audio_forget, gt_one_hot_forget)
    raw_a4 = net_a4(vec_audio_forget).mean()

    # ── a5: UKR video retain ──────────────────────────────────────────────────
    teacher_video_retain = func_video_only(out_dr_ori, retain_labels)
    vec_video_retain = get_diff_vectors(out_dr_un["video_logits"], teacher_video_retain, gt_one_hot_retain)
    raw_a5 = net_a5(vec_video_retain).mean()

    # ── a6: UKR audio retain (CRITICAL - audio learning from video) ──────────
    teacher_audio_retain = func_video_only(out_dr_ori, retain_labels)
    vec_audio_retain = get_diff_vectors(out_dr_un["audio_logits"], teacher_audio_retain, gt_one_hot_retain)
    raw_a6 = net_a6(vec_audio_retain).mean()

    # ── Normalise and boost audio learning (a6) ────────────────────────────────
    scores = torch.stack([raw_a1, raw_a2, raw_a3, raw_a4, raw_a5, raw_a6])
    weights = F.softmax(scores, dim=0) * 6
    # weights[5] = weights[5] * 2.0  # Boost a6 (audio retain - most important)
    
    return weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP: compute total loss with 6 dynamic alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
                             net_a1, net_a2, net_a3, net_a4, net_a5, net_a6,
                             batch_df, batch_dr,
                             training: bool = True):
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
        out_df_ori = model_ori(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori = model_ori(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

        perm = torch.randperm(batch_df["spectrogram"].size(0))
        rand_spec = batch_df["spectrogram"][perm]
        out_ori_random = model_ori(batch_df["video"], rand_spec, return_intermediate=True)

    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    a1, a2, a3, a4, a5, a6 = compute_dynamic_alphas(
        net_a1, net_a2, net_a3, net_a4, net_a5, net_a6,
        out_df_un, out_dr_un,
        out_df_ori, out_dr_ori,
        forget_labels, retain_labels,
        gt_one_hot_forget, gt_one_hot_retain,
    )

    # ── INDIVIDUAL LOSS TERMS (one alpha per component) ───────────────────────
    
    # a1: Forget fusion with uniform distribution
    loss_md = Uniform_MD_loss(out_df_un["fusion_logits"], forget_labels, T, reduction="batchmean")
    
    # a2: Retain fusion protection (don't re-learn forget)
    loss_mkr_fusion = MKR_loss(
        func_max_retain(out_dr_ori, retain_labels), 
        out_dr_un["fusion_logits"], 
        T, forget_labels, reduction="batchmean"
    )
    
    # a3: Video learn from best on forget set
    loss_ukr_video_forget = UKR_loss(
        func_max(out_df_ori, forget_labels), 
        out_df_un["video_logits"], 
        T, reduction="batchmean"
    )
    
    # a4: Audio learn from best on forget set
    loss_ukr_audio_forget = UKR_loss(
        func_max(out_df_ori, forget_labels),
        out_df_un["audio_logits"], 
        T, reduction="batchmean"
    )
    
    # a5: Video learn from video on retain set
    loss_ukr_video_retain = UKR_loss(
        func_video_only(out_dr_ori, retain_labels),
        out_dr_un["video_logits"], 
        T, reduction="batchmean"
    )
    
    # a6: Audio learn from video on retain set (CRITICAL)
    loss_ukr_audio_retain = UKR_loss(
        func_video_only(out_dr_ori, retain_labels),
        out_dr_un["audio_logits"], 
        T, reduction="batchmean"
    )
    
    total_loss = (a1*loss_md) + (a2*loss_mkr_fusion) + (a3*loss_ukr_video_forget) + \
                 (a4*loss_ukr_audio_forget) + (a5*loss_ukr_video_retain) + (a6*loss_ukr_audio_retain)

    return {
        "train_loss": total_loss,
        "loss_md": loss_md.detach(),
        "loss_mkr": loss_mkr_fusion.detach(),
        "loss_video_forget": loss_ukr_video_forget.detach(),
        "loss_audio_forget": loss_ukr_audio_forget.detach(),
        "loss_video_retain": loss_ukr_video_retain.detach(),
        "loss_audio_retain": loss_ukr_audio_retain.detach(),
        "a1": a1.item(),
        "a2": a2.item(),
        "a3": a3.item(),
        "a4": a4.item(),
        "a5": a5.item(),
        "a6": a6.item(),
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP with early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
best_nets_state = {}

print(f"Starting 6-alpha unlearning for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  6 Alphas: MD | MKR | Video_forget | Audio_forget | Video_retain | Audio_retain")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}")
print(f"  Checkpoint → {CHECKPOINT_PATH}\n")

for epoch in range(EPOCHS):
    model_unlearn.train()
    for net in [net_a1, net_a2, net_a3, net_a4, net_a5, net_a6]:
        net.train()

    running_train_loss = 0.0
    n_train_batches = 0
    total_train_samples = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        batch_size_df = batch_df["video"].size(0)
        batch_size_dr = batch_dr["video"].size(0)
        total_train_samples += (batch_size_df + batch_size_dr)
        
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            net_a1, net_a2, net_a3, net_a4, net_a5, net_a6,
            batch_df, batch_dr,
            training=True,
        )

        out["train_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_unlearn.parameters()) +
            list(net_a1.parameters()) +
            list(net_a2.parameters()) +
            list(net_a3.parameters()) +
            list(net_a4.parameters()) +
            list(net_a5.parameters()) +
            list(net_a6.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    model_unlearn.eval()
    for net in [net_a1, net_a2, net_a3, net_a4, net_a5, net_a6]:
        net.eval()

    running_val_loss = 0.0
    n_val_batches = 0
    total_val_samples = 0
    last_out = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            batch_size_df = batch_df["video"].size(0)
            batch_size_dr = batch_dr["video"].size(0)
            total_val_samples += (batch_size_df + batch_size_dr)
            
            out = compute_unlearning_loss(
                model_ori, model_unlearn,
                net_a1, net_a2, net_a3, net_a4, net_a5, net_a6,
                batch_df, batch_dr,
                training=False,
            )
            running_val_loss += out["train_loss"].item()
            n_val_batches += 1
            last_out = out

    avg_val_loss = running_val_loss / max(n_val_batches, 1)

    if last_out is not None:
        print(
            f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} ({total_train_samples} samples in {n_train_batches} batches) | Val {avg_val_loss:.4f} ({total_val_samples} samples in {n_val_batches} batches) | "
            f"a1={last_out['a1']:.2f} a2={last_out['a2']:.2f} a3={last_out['a3']:.2f} "
            f"a4={last_out['a4']:.2f} a5={last_out['a5']:.2f} a6={last_out['a6']:.2f}"
        )
    else:
        print(f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())
        best_nets_state = {
            "net_a1": deepcopy(net_a1.state_dict()),
            "net_a2": deepcopy(net_a2.state_dict()),
            "net_a3": deepcopy(net_a3.state_dict()),
            "net_a4": deepcopy(net_a4.state_dict()),
            "net_a5": deepcopy(net_a5.state_dict()),
            "net_a6": deepcopy(net_a6.state_dict()),
        }

        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        torch.save({
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "model_state_dict": model_unlearn.state_dict(),
            "net_a1_state_dict": net_a1.state_dict(),
            "net_a2_state_dict": net_a2.state_dict(),
            "net_a3_state_dict": net_a3.state_dict(),
            "net_a4_state_dict": net_a4.state_dict(),
            "net_a5_state_dict": net_a5.state_dict(),
            "net_a6_state_dict": net_a6.state_dict(),
            "optimizer_state": optimizer.state_dict(),
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
    net_a5.load_state_dict(best_nets_state["net_a5"])
    net_a6.load_state_dict(best_nets_state["net_a6"])

    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    torch.save({
        "epoch": "best",
        "val_loss": best_val_loss,
        "model_state_dict": model_unlearn.state_dict(),
        "net_a1_state_dict": net_a1.state_dict(),
        "net_a2_state_dict": net_a2.state_dict(),
        "net_a3_state_dict": net_a3.state_dict(),
        "net_a4_state_dict": net_a4.state_dict(),
        "net_a5_state_dict": net_a5.state_dict(),
        "net_a6_state_dict": net_a6.state_dict(),
    }, CHECKPOINT_PATH)
    print("Restored best model weights.")

print(f"\nUnlearning complete.")
print(f"  Unlearned model : {UNLEARNED_MODEL_PATH}")
print(f"  Full checkpoint : {CHECKPOINT_PATH}")
