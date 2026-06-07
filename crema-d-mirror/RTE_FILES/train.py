"""
Train CREMA-D Dual-ResNet50 Multimodal Model
=============================================
Trains the CremaMultimodalModel on the CREMA-D audio-video dataset.

Three simultaneous classification branches:
  - image-only branch
  - audio-only branch
  - fusion (image+audio) branch

Loss = CE(image_logits) + CE(audio_logits) + CE(fusion_logits)

Usage:
    # From ADVANCE_Unlearning/ directory:
    python scripts/train.py

    # Or from newDirauth2/ root:
    python -m ADVANCE_Unlearning.scripts.train
"""

import os
import sys
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running directly from the ADVANCE_Unlearning directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_base_splits
from src.model   import CremaMultimodalModel
from src.labels  import NUM_CLASSES

from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 1e-4
PATIENCE   = 10          # Early stopping patience
VAL_RATIO  = 0.2
SEED       = 42

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/rte", "crema_trained_05.pth")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ─── Early Stopping ──────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0


# ─── Helpers ─────────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return (logits.argmax(dim=1) == labels).sum().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n_correct_vid = n_correct_aud = n_correct_fus = n_total = 0
    total_loss = 0.0

    for batch in loader:
        video  = batch["video"].to(device)
        spec   = batch["spectrogram"].to(device)
        labels = batch["label"].to(device)

        out = model(video, spec, return_intermediate=True)
        loss = (
            F.cross_entropy(out["fusion_logits"], labels) +
            F.cross_entropy(out["video_logits"],  labels) +
            F.cross_entropy(out["audio_logits"],  labels)
        )
        total_loss      += loss.item()
        n_correct_vid   += accuracy(out["video_logits"],  labels)
        n_correct_aud   += accuracy(out["audio_logits"],  labels)
        n_correct_fus   += accuracy(out["fusion_logits"], labels)
        n_total         += labels.size(0)

    return (
        n_correct_vid / n_total,
        n_correct_aud / n_total,
        n_correct_fus / n_total,
        total_loss    / len(loader),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Num classes: {NUM_CLASSES}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building dataset (spectrogram cache may take a few minutes on first run)...")
    train_ds, val_ds, test_ds = get_base_splits(seed=SEED)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = CremaMultimodalModel(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    stopper   = EarlyStopping(patience=PATIENCE)

    best_val_loss = float("inf")

    # ── Training Loop ──────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in loop:
            video  = batch["video"].to(DEVICE)
            spec   = batch["spectrogram"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            out  = model(video, spec, return_intermediate=True)
            loss = (
                F.cross_entropy(out["fusion_logits"], labels) +
                F.cross_entropy(out["video_logits"],  labels) +
                F.cross_entropy(out["audio_logits"],  labels)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        acc_vid, acc_aud, acc_fus, val_loss = evaluate(model, val_loader, DEVICE)

        print(
            f"\nEpoch {epoch:02d} | "
            f"TrainLoss {train_loss/len(train_loader):.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"Vid {acc_vid*100:.1f}% | "
            f"Aud {acc_aud*100:.1f}% | "
            f"Fusion {acc_fus*100:.1f}%"
        )
        
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved → {MODEL_PATH}")

        stopper(val_loss)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best model: {MODEL_PATH}")


if __name__ == "__main__":
    _t_start = time.perf_counter()
    main()
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (train.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")
