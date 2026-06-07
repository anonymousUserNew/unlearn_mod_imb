import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import DcaseMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "dcase_unlearned_perf_aware.pth")
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "models", "dcase_chk_perf_aware.pth")

BATCH_SIZE = 16
LR         = 1e-5
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 2.0           

C = NUM_CLASSES             

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
model_ori = DcaseMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

# Freeze original (teacher)
for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model_unlearn.parameters(), lr=LR)

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

# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE-AWARE ALPHA COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_accuracies(model, forget_loader, retain_loader):
    """
    Compute actual accuracies on validation set.
    Returns dict of accuracies for all 6 components.
    """
    model.eval()
    accuracies = {}
    
    with torch.no_grad():
        # Forget set accuracies
        correct_video_f = correct_audio_f = correct_fusion_f = total_f = 0
        for batch in forget_loader:
            video = batch["video"].to(DEVICE)
            spec = batch["spectrogram"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            
            out = model(video, spec, return_intermediate=True)
            correct_video_f += (out["video_logits"].argmax(1) == label).sum().item()
            correct_audio_f += (out["audio_logits"].argmax(1) == label).sum().item()
            correct_fusion_f += (out["fusion_logits"].argmax(1) == label).sum().item()
            total_f += label.size(0)
        
        accuracies["video_forget"]  = correct_video_f / total_f if total_f > 0 else 0.0
        accuracies["audio_forget"]  = correct_audio_f / total_f if total_f > 0 else 0.0
        accuracies["fusion_forget"] = correct_fusion_f / total_f if total_f > 0 else 0.0
        
        # Retain set accuracies
        correct_video_r = correct_audio_r = correct_fusion_r = total_r = 0
        for batch in retain_loader:
            video = batch["video"].to(DEVICE)
            spec = batch["spectrogram"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            
            out = model(video, spec, return_intermediate=True)
            correct_video_r += (out["video_logits"].argmax(1) == label).sum().item()
            correct_audio_r += (out["audio_logits"].argmax(1) == label).sum().item()
            correct_fusion_r += (out["fusion_logits"].argmax(1) == label).sum().item()
            total_r += label.size(0)
        
        accuracies["video_retain"]  = correct_video_r / total_r if total_r > 0 else 0.0
        accuracies["audio_retain"]  = correct_audio_r / total_r if total_r > 0 else 0.0
        accuracies["fusion_retain"] = correct_fusion_r / total_r if total_r > 0 else 0.0
    
    return accuracies

def compute_adaptive_alphas(accuracies, epoch):
    """
    Compute 6 alphas based on performance gaps from targets.
    
    Alpha logic:
    - a1 (MD):           gap = fusion_forget (want 0%, so gap is actual)
    - a2 (MKR):          gap = 1 - fusion_retain (protect it)
    - a3 (Video forget): gap = 1 - video_forget (maintain)
    - a4 (Audio forget): gap = 1 - audio_forget (learn from video)
    - a5 (Video retain): gap = 1 - video_retain (protect from degradation)
    - a6 (Audio retain): gap = 1 - audio_retain (CRITICAL - main bottleneck)
    
    Larger gaps = higher alpha (more weight to fix the bottleneck)
    """
    
    # Define target accuracies
    targets = {
        "video_forget":  0.95,   # Video should stay high on forget
        "audio_forget":  0.75,   # Audio should learn from video on forget
        "fusion_forget": 0.00,   # Fusion should be completely unlearned
        "video_retain":  0.90,   # Video should maintain on retain
        "audio_retain":  0.45,   # Audio should learn from video on retain
        "fusion_retain": 0.85,   # Fusion should be maintained on retain
    }
    
    # Compute gaps (how far from target)
    # For fusion_forget: gap is the actual accuracy (closer to 0 is better, so gap = actual)
    # For others: gap is target - actual (positive when underperforming)
    gaps = [
        accuracies["fusion_forget"],                      # a1: want 0%, so gap = actual acc
        max(0, targets["fusion_retain"] - accuracies["fusion_retain"]),  # a2
        max(0, targets["video_forget"] - accuracies["video_forget"]),    # a3
        max(0, targets["audio_forget"] - accuracies["audio_forget"]),    # a4
        max(0, targets["video_retain"] - accuracies["video_retain"]),    # a5
        max(0, targets["audio_retain"] - accuracies["audio_retain"]),    # a6
    ]
    
    # Normalize gaps to create probabilities
    total_gap = sum(gaps) + 1e-8
    alpha_probs = torch.tensor([g / total_gap for g in gaps], dtype=torch.float32)
    
    # Scale to roughly sum to 6 (similar to softmax * 6 approach)
    alphas = alpha_probs * 6.0
    
    # Extra boost to a6 (audio retain) if it's the biggest bottleneck
    # But let the network decide based on actual gaps, not hard coding
    if gaps[5] > max(gaps[3], gaps[4]):  # If audio_retain gap is largest
        alphas[5] = alphas[5] * 1.5
    
    return alphas.tolist(), gaps, targets

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP: compute total loss with adaptive alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
                             batch_df, batch_dr,
                             alphas):
    for k in batch_df:
        if isinstance(batch_df[k], torch.Tensor):
            batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        if isinstance(batch_dr[k], torch.Tensor):
            batch_dr[k] = batch_dr[k].to(DEVICE)

    forget_labels  = batch_df["label"]
    retain_labels  = batch_dr["label"]

    with torch.no_grad():
        out_df_ori = model_ori(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori = model_ori(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    a1, a2, a3, a4, a5, a6 = alphas

    # ── INDIVIDUAL LOSS TERMS (one alpha per component) ───────────────────────
    
    # a1: Forget fusion with uniform distribution (target 0%)
    loss_md = Uniform_MD_loss(out_df_un["fusion_logits"], forget_labels, T, reduction="batchmean")
    
    # a2: Retain fusion protection (maintain high accuracy)
    loss_mkr_fusion = UKR_loss(
        func_max_retain(out_dr_ori, retain_labels), 
        out_dr_un["fusion_logits"], 
        T, reduction="batchmean"
    )
    
    # a3: Video learn from best on forget set (maintain video quality)
    loss_ukr_video_forget = UKR_loss(
        func_max(out_df_ori, forget_labels), 
        out_df_un["video_logits"], 
        T, reduction="batchmean"
    )
    
    # a4: Audio learn from best on forget set (cross-modal learning)
    loss_ukr_audio_forget = UKR_loss(
        func_max(out_df_ori, forget_labels), 
        out_df_un["audio_logits"], 
        T, reduction="batchmean"
    )
    
    # a5: Video learn from video on retain set (maintain quality)
    loss_ukr_video_retain = UKR_loss(
        func_video_only(out_dr_ori), 
        out_dr_un["video_logits"], 
        T, reduction="batchmean"
    )
    
    # a6: Audio learn from video on retain set (CRITICAL cross-modal)
    loss_ukr_audio_retain = UKR_loss(
        func_video_only(out_dr_ori), 
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
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "a5": a5,
        "a6": a6,
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP with performance-aware alphas
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
alpha_history = []

print(f"Starting performance-aware unlearning for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  Alphas adapt based on validation accuracy gaps (not hardcoded).")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}\n")

for epoch in range(EPOCHS):
    model_unlearn.train()
    
    # Compute adaptive alphas based on current validation performance
    accuracies = compute_accuracies(model_unlearn, forget_val_loader, retain_val_loader)
    alphas, gaps, targets = compute_adaptive_alphas(accuracies, epoch)
    alpha_history.append(alphas)

    running_train_loss = 0.0
    n_train_batches = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            batch_df, batch_dr,
            alphas,
        )

        out["train_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model_unlearn.parameters(), max_norm=1.0)
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    model_unlearn.eval()

    running_val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori, model_unlearn,
                batch_df, batch_dr,
                alphas,
            )
            running_val_loss += out["train_loss"].item()
            n_val_batches += 1
            last_out = out

    avg_val_loss = running_val_loss / max(n_val_batches, 1)

    # Print epoch summary with accuracies and alphas
    print(
        f"Epoch {epoch:3d} | Loss {avg_val_loss:.4f} | "
        f"V_f={accuracies['video_forget']:.2%} A_f={accuracies['audio_forget']:.2%} F_f={accuracies['fusion_forget']:.2%} | "
        f"V_r={accuracies['video_retain']:.2%} A_r={accuracies['audio_retain']:.2%} F_r={accuracies['fusion_retain']:.2%} | "
        f"a=[{alphas[0]:.2f} {alphas[1]:.2f} {alphas[2]:.2f} {alphas[3]:.2f} {alphas[4]:.2f} {alphas[5]:.2f}]"
    )
    print(f"         Gaps: F_f={gaps[0]:.2f} F_r={gaps[1]:.2f} V_f={gaps[2]:.2f} A_f={gaps[3]:.2f} V_r={gaps[4]:.2f} A_r={gaps[5]:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())

        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        torch.save({
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "model_state_dict": model_unlearn.state_dict(),
            "accuracies": accuracies,
            "alphas": alphas,
            "optimizer_state": optimizer.state_dict(),
            "alpha_history": alpha_history,
        }, CHECKPOINT_PATH)
        print(f"  ✓ Best val loss {best_val_loss:.4f}. Model saved.")
    else:
        patience_counter += 1
        print(f"  ✗ No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    print("Restored and saved best model weights.")

print(f"\nUnlearning complete.")
print(f"  Model saved: {UNLEARNED_MODEL_PATH}")
print(f"  Checkpoint  : {CHECKPOINT_PATH}")
