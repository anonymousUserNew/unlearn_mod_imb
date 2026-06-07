import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from src.dataset import create_train_val_split
from src.model_new_r import MultimodalFoodClassifier

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Oracle training uses only the full retain dataset (everything except cup_cakes)
TRAIN_CSV = "/home/team2/Unlearning/Food101-files/data/annotations/train_titles_retain_full.csv"
IMAGE_ROOT = "/home/team2/Unlearning/Food101-files/data/images/images/train"

BATCH_SIZE = 16            # safer for BERT + ResNet
MAX_EPOCHS = 40
VAL_RATIO = 0.2
PATIENCE = 10
NUM_CLASSES = 101

# Save path for the oracle model
SAVE_DIR = "/home/team2/Unlearning/Food101-files/models"

os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "oracle_model.pth")

# --------------------------------------------------
# EARLY STOPPING
# --------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

# --------------------------------------------------
# MAIN TRAINING SCRIPT
# --------------------------------------------------
def main():
    print(f"Creating train/validation split using {TRAIN_CSV}...")
    train_dataset, val_dataset, labels = create_train_val_split(
        TRAIN_CSV,
        IMAGE_ROOT,
        val_ratio=VAL_RATIO,
        seed=42
    )

    print(f"Number of classes: {len(labels)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print("Initializing model...")
    model = MultimodalFoodClassifier(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # --------------------------------------------------
    # LOSS & OPTIMIZER (Layer-wise LR)
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": model.image_encoder.parameters(), "lr": 1e-4},
        {"params": model.text_proj.parameters(), "lr": 1e-3},
        {"params": model.image_proj.parameters(), "lr": 1e-3},
        {"params": model.fusion.parameters(), "lr": 1e-3},
        {"params": model.fusion_classifier.parameters(), "lr": 1e-3},
        {"params": model.text_classifier.parameters(), "lr": 1e-3},
        {"params": model.image_classifier.parameters(), "lr": 1e-3},
    ])

    early_stopper = EarlyStopping(patience=PATIENCE)

    print("Starting Oracle training...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            image = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_tensor = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_image=True,
                use_text=True,
                return_intermediate=True
            )

            fusion_logits, text_logits, image_logits = outputs["fusion_logits"], outputs["text_logits"], outputs["image_logits"]

            loss = criterion(fusion_logits, labels_tensor)+ criterion(text_logits, labels_tensor)+ criterion(image_logits, labels_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                image = batch["image"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels_tensor = batch["label"].to(DEVICE)

                outputs = model(
                    image=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_image=True,
                    use_text=True,
                    return_intermediate=True
                )

                fusion_logits, text_logits, image_logits = outputs["fusion_logits"], outputs["text_logits"], outputs["image_logits"]

                loss = criterion(fusion_logits, labels_tensor)+ criterion(text_logits, labels_tensor)+ criterion(image_logits, labels_tensor)
                val_loss += loss.item()

                preds = torch.argmax(fusion_logits, dim=1)
                correct += (preds == labels_tensor).sum().item()
                total += labels_tensor.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} | "
            f"Val Acc = {val_acc:.4f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if early_stopper.step(val_loss):
            torch.save(model.state_dict(), SAVE_PATH)
            print("  → Best oracle model saved.")

        if early_stopper.should_stop():
            print("\nEarly stopping triggered.")
            break

    print("\nOracle Training complete.")
    print(f"Best oracle model saved at: {SAVE_PATH}")

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
