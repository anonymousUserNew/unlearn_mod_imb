"""
Train ADVANCE Dual-ResNet50 Multimodal Model - Retain Split Only
================================================================
Trains the AdvanceMultimodalModel from scratch ONLY on the retain classes.
This serves as an Oracle/Gold Standard for unlearning, allowing a precise
comparison of whether the unlearnt model accurately mimics a model that has
never seen the forget data.

Loss = CE(image_logits) + CE(audio_logits) + CE(fusion_logits)

Usage:
    python scripts/train_retain.py
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running directly from the project root or scripts directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import RetainDataset, FORGET_CLASS
from src.model   import AdvanceMultimodalModel
from src.labels  import NUM_CLASSES, ADVANCE_CLASSES

from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 1e-4
PATIENCE   = 10          # Early stopping patience
SEED       = 42

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models_rte", "advance_trained_retain_only.pth")
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
    n_correct_img = n_correct_aud = n_correct_fus = n_total = 0
    total_loss = 0.0

    for batch in loader:
        image  = batch["image"].to(device)
        spec   = batch["spectrogram"].to(device)
        labels = batch["label"].to(device)

        out = model(image, spec, return_intermediate=True)
        loss = (
            F.cross_entropy(out["fusion_logits"], labels) +
            F.cross_entropy(out["image_logits"],  labels) +
            F.cross_entropy(out["audio_logits"],  labels)
        )
        total_loss      += loss.item()
        n_correct_img   += accuracy(out["image_logits"],  labels)
        n_correct_aud   += accuracy(out["audio_logits"],  labels)
        n_correct_fus   += accuracy(out["fusion_logits"], labels)
        n_total         += labels.size(0)

    # Prevent division by zero if loader is empty
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        n_correct_img / n_total,
        n_correct_aud / n_total,
        n_correct_fus / n_total,
        total_loss    / len(loader),
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Total Num Classes in Architecture: {NUM_CLASSES}")
    print(f"Forget Class to Filter Out: '{FORGET_CLASS}'\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building retain-only dataset ...")

    # We load ONLY the retain dataset, and split it into train/val
    retain_full = RetainDataset(forget_class=FORGET_CLASS)
    n = len(retain_full)
    n_val = int(n * 0.2)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(SEED)
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(retain_full, [n_train, n_val], generator=gen)

    print(f"  Retain Train: {len(train_ds)}  |  Retain Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    # Note: even though we're training only on retain classes, we instantiate
    # the model with all NUM_CLASSES so its architecture exactly matches the
    # unlearned model. It simply won't predict the forget class.
    model     = AdvanceMultimodalModel(num_classes=NUM_CLASSES).to(DEVICE)
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
            image  = batch["image"].to(DEVICE)
            spec   = batch["spectrogram"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            out = model(image, spec, return_intermediate=True)

            # Forward labels to standard cross entropy.
            # Even though there are NUM_CLASSES logits, the ground truth
            # labels will never be FORGET_CLASS index, so it trains naturally.
            loss = (
                F.cross_entropy(out["fusion_logits"], labels) +
                F.cross_entropy(out["image_logits"],  labels) +
                F.cross_entropy(out["audio_logits"],  labels)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        acc_img, acc_aud, acc_fus, val_loss = evaluate(model, val_loader, DEVICE)

        print(
            f"\nEpoch {epoch:02d} | "
            f"TrainLoss {train_loss/len(train_loader):.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"Img {acc_img*100:.1f}% | "
            f"Aud {acc_aud*100:.1f}% | "
            f"Fusion {acc_fus*100:.1f}%"
        )

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best retain-only model saved → {MODEL_PATH}")

        stopper(val_loss)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best retain-only model: {MODEL_PATH}")


if __name__ == "__main__":
    _t_start = time.perf_counter()
    main()
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (train_retain.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")
