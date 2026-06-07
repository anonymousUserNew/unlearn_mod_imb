import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CremaDataset, discover_pairs, FORGET_CLASS, get_full_dataset, get_base_splits, get_forget_splits, get_retain_splits
from src.model   import CremaMultimodalModel
from src.labels  import CREMA_CLASSES, NUM_CLASSES


# --------------------------------------------------
# CONFIG  ← edit these to switch model / split
# --------------------------------------------------
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
# NUM_CLASSES is imported from src.labels (6 for CREMA-D)

# Which model to evaluate:
MODEL_PATH = "/home/team2/Unlearning/crema-d-mirror/models/unimodal_unlearn/crema_unlearned_dtd.pth"
# MODEL_PATH = "/home/team2/Unlearning/crema-d-mirror/models/crema_trained_05.pth"

# Which data split to evaluate on:
#   "val"    → 10% held-out validation set ← USE THIS for honest evaluation
#              (same reproducible split as training, seed=42)
#   "full"   → entire dataset (includes training data — numbers will be inflated)
#   "forget" → only the forget class (default: HAP)
#   "retain" → all classes except the forget class
#   "train"  → 80% training portion (sanity/overfit check only)

#DATA_SPLIT = "val"
# DATA_SPLIT = "full"
DATA_SPLIT = "retain"
#DATA_SPLIT = "val"

# Where to save outputs (update this to match MODEL_PATH and DATA_SPLIT):
OUTPUT_DIR = "/home/team2/Unlearning/crema-d-mirror/outputs/unimodal_eval/dtd/retain"
# OUTPUT_DIR = "/home/team2/Unlearning/crema-d-mirror/outputs/unimodal_eval/retain"
# OUTPUT_DIR = "/home/team2/Unlearning/crema-d-mirror/outputs/unimodal_eval/val"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
# Reproducible val/train split — must match train.py (seed=42)
SPLIT_SEED = 42

print(f"Loading dataset split: '{DATA_SPLIT}' ...")

all_pairs = discover_pairs()
full_dataset = CremaDataset(all_pairs, is_train=False)

if DATA_SPLIT in ["train", "val", "test"]:
    train_ds, val_ds, test_ds = get_base_splits(seed=SPLIT_SEED)
    if DATA_SPLIT == "train": dataset = train_ds
    elif DATA_SPLIT == "val": dataset = val_ds
    else: dataset = test_ds
    print(f"  (Using base {DATA_SPLIT} split)")
elif DATA_SPLIT == "full":
    dataset = full_dataset
elif DATA_SPLIT == "forget":
    _, _, dataset = get_forget_splits(seed=SPLIT_SEED)
    print("  (Evaluating on the test split for forget class)")
elif DATA_SPLIT == "retain":
    _, _, dataset = get_retain_splits(seed=SPLIT_SEED)
    print("  (Evaluating on the test split for retain classes)")
else:
    raise ValueError(f"Unknown DATA_SPLIT '{DATA_SPLIT}'. Choose: test, val, full, forget, retain, train")

print(f"Dataset size: {len(dataset)} samples")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

class_names = CREMA_CLASSES  # list of 6 string names


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print(f"Loading model from: {MODEL_PATH}")
model = CremaMultimodalModel(num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def per_class_accuracy(y_true, y_pred, num_classes):
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    return [
        (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]


def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar(format="%.0f%%")
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, classes, fontsize=8)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate(branch="fusion", tag="fusion"):
    """
    branch: 'video' | 'audio' | 'fusion'
    tag   : prefix for saved files
    """
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            video  = batch["video"].to(DEVICE)
            spec   = batch["spectrogram"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            out = model(video, spec, return_intermediate=True)

            if branch == "video":
                logits = out["video_logits"]
            elif branch == "audio":
                logits = out["audio_logits"]
            else:
                logits = out["fusion_logits"]

            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    metrics_df = pd.DataFrame({
        "Class"    : class_names,
        "Precision": precision,
        "Recall"   : recall,
        "F1-Score" : f1,
        "Support"  : support,
    })

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm       = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm.astype(float) / row_sums * 100

    # ── Per-Class Accuracy ────────────────────────────────────────────────────
    class_acc = per_class_accuracy(y_true, y_pred, NUM_CLASSES)
    metrics_df["Accuracy%"] = class_acc

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics_df.to_csv(f"{OUTPUT_DIR}/{tag}_classwise_metrics.csv", index=False)

    np.savetxt(
        f"{OUTPUT_DIR}/{tag}_confusion_matrix_normalized.csv",
        cm_norm, delimiter=",", fmt="%.2f"
    )

    with open(f"{OUTPUT_DIR}/{tag}_summary.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Split: {DATA_SPLIT}\n")
        f.write(f"Branch: {branch}\n")
        f.write(f"Dataset: CREMA-D (6 emotion classes)\n")

    plot_confusion_matrix(
        cm_norm, class_names,
        f"{branch.capitalize()} Branch Confusion Matrix (%) — {DATA_SPLIT}",
        f"{OUTPUT_DIR}/{tag}_confusion_matrix.png"
    )

    print(f"[{tag:>10}]  Overall Accuracy: {accuracy*100:.2f}%")
    return cm_norm, metrics_df, accuracy


# --------------------------------------------------
# RUN — all 3 branches
# --------------------------------------------------
print(f"\nEvaluating VIDEO-ONLY branch ...")
cm_video, df_video, acc_video = evaluate(branch="video", tag="video_only")

print(f"\nEvaluating AUDIO-ONLY branch ...")
cm_audio, df_audio, acc_audio = evaluate(branch="audio", tag="audio_only")

print(f"\nEvaluating MULTIMODAL (fusion) branch ...")
cm_fusion, df_fusion, acc_fusion = evaluate(branch="fusion", tag="multimodal")


# --------------------------------------------------
# PER-CLASS COMPARISON TABLE
# --------------------------------------------------
comparison_df = pd.DataFrame({
    "Class"             : class_names,
    "Video Accuracy%"   : df_video["Accuracy%"],
    "Audio Accuracy%"   : df_audio["Accuracy%"],
    "Fusion Accuracy%"  : df_fusion["Accuracy%"],
})
comparison_df.to_csv(f"{OUTPUT_DIR}/per_class_accuracy_comparison.csv", index=False)

print(f"\nPer-class accuracy comparison saved.")
print(f"\nEvaluation complete. All results in: {OUTPUT_DIR}")
