import os
import sys
import time
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

_t_start = time.perf_counter()

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models_rte", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models_rte", "dcase_unlearned_embed.pth")  # v4
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "models_rte", "dcase_unlearned_embed_chk.pth")

BATCH_SIZE = 8
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
mse=nn.MSELoss()

model_ori = DcaseMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

# Freeze original (teacher)
for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    list(model_unlearn.parameters()),
    lr=LR,
)

# ──────────────────────────────────────────────────────────────────────────────
# CORE STEP:  compute total loss with dynamic alphas
# ──────────────────────────────────────────────────────────────────────────────
def compute_unlearning_loss(model_ori, model_unlearn,
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
        perm       = torch.randperm(batch_dr["spectrogram"].size(0))
        rand_spec  = batch_dr["spectrogram"][perm]
        out_ori_random = model_ori(batch_dr["video"], rand_spec, return_intermediate=True)

    # ── Student forward ───────────────────────────────────────────────────────
    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    # ── Individual losses ─────────────────────────────────────────────────────

    # L_MD: push fusion branch toward UNIFORM over the 12 non-forget classes.
    # Previously used a random-pair teacher, which caused the fusion branch to
    # confidently predict whatever class that random pair resembled.  A uniform
    # target directly encodes the desired outcome: max uncertainty on forget data.
    loss_md = mse(out_df_un["fused_emb"], out_ori_random["fused_emb"])

    # L_MKR: all 3 branches on retain data (averaged)
    loss_mkr = mse(out_dr_un["fused_emb"],out_dr_ori["fused_emb"])

    # L_UKR (dynamic alpha): KD from original teacher on forget data.
    # Keeps unimodal distributions close to original model.
    loss_ukr = 0.5 * (
        mse(out_df_un["vid_emb"], out_df_ori["vid_emb"]) +
        mse(out_df_un["aud_emb"], out_df_ori["aud_emb"])
    )

    # L_UNI_CE (fixed weight, outside alpha system): direct cross-entropy on
    # image + audio branches for the forget class.  Ensures unimodal branches
    # always receive a strong gradient to keep classifying 'residential'
    # correctly, regardless of what a3 does.
    # loss_uni_ce = 0.5 * (
    #     F.cross_entropy(out_df_un["image_logits"], forget_labels) +
    #     F.cross_entropy(out_df_un["audio_logits"], forget_labels)
    # )

    total_loss = (loss_md) + (loss_mkr) + (loss_ukr)

    return {
        "train_loss" : total_loss,
        "loss_md"    : loss_md.detach(),
        "loss_multi" : loss_mkr.detach(),
        "loss_uni"   : loss_ukr.detach()
    }

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  with early stopping
# ──────────────────────────────────────────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
best_model_state = None

print(f"Starting dynamic-alpha unlearning for {EPOCHS} epochs (patience={PATIENCE}).")
print(f"  Teacher : {TRAINED_MODEL_PATH}")
print(f"  Student → {UNLEARNED_MODEL_PATH}")

for epoch in range(EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────────
    model_unlearn.train()

    running_train_loss = 0.0
    n_train_batches    = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori, model_unlearn,
            batch_df, batch_dr,
            training=True,
        )

        out["train_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_unlearn.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        running_train_loss += out["train_loss"].item()
        n_train_batches    += 1

    avg_train_loss = running_train_loss / max(n_train_batches, 1)

    # ── Validate ───────────────────────────────────────────────────────────────
    model_unlearn.eval()

    running_val_loss = 0.0
    n_val_batches    = 0
    last_out         = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori, model_unlearn,
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
        )
    else:
        print(f"Epoch {epoch:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

    # ── Early Stopping ─────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        best_model_state = deepcopy(model_unlearn.state_dict())
        # ── Save both artefacts immediately ───────────────────────────────────
        # 1. Just the unlearned model weights (drop-in for eval.py)
        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)

        # 2. Full checkpoint (model + WeightNets + metadata)
        torch.save({
            "epoch"            : epoch,
            "val_loss"         : avg_val_loss,
            "model_state_dict" : model_unlearn.state_dict(),
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

    # Overwrite with (definitely) best state
    torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
    torch.save({
        "epoch"            : "best",
        "val_loss"         : best_val_loss,
        "model_state_dict" : model_unlearn.state_dict()
    }, CHECKPOINT_PATH)
    print("Restored best model weights.")

print(f"\nUnlearning complete.")
print(f"  Unlearned model : {UNLEARNED_MODEL_PATH}")
print(f"  Full checkpoint : {CHECKPOINT_PATH}")

_t_end  = time.perf_counter()
_elapsed = _t_end - _t_start
_h, _rem = divmod(int(_elapsed), 3600)
_m, _s   = divmod(_rem, 60)
print(f"\n{'='*60}")
print(f"  Runtime (unlearn_4losses_01.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
print(f"{'='*60}")