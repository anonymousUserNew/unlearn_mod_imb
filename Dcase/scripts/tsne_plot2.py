"""
tsne_plot.py
============
Generate t-SNE scatter plots for the three embedding types produced by
DcaseMultimodalModel:
  • fused_emb  (512-d)
  • vid_emb    (512-d)
  • aud_emb    (512-d)

The forget class is always coloured RED; all other classes get distinct colours.

Usage
-----
Edit the CONFIG block below and run:
    python scripts/tsne_plot.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# ── project root on the path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_base_splits, FORGET_CLASS
from src.model   import DcaseMultimodalModel
from src.labels  import DCASE_CLASSES, NUM_CLASSES, LABEL_MAP

# =============================================================================
# CONFIG  ← edit these
# =============================================================================
MODEL_PATH  = "/home/team2/Unlearning/Dcase/models/dcase_trained.pth"
OUTPUT_DIR  = "/home/team2/Unlearning/Dcase/outputs/tsne2/trained"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
TSNE_PERPLEXITY = 50
TSNE_ITER       = 1000
RANDOM_STATE    = 42
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

FORGET_IDX = LABEL_MAP[FORGET_CLASS]   # integer class index for the forget class

# ── colour palette ────────────────────────────────────────────────────────────
# We need NUM_CLASSES colours total.
# Index FORGET_IDX → red (#e83030).
# All others → distinct high-contrast colours.
# ── colour palette ────────────────────────────────────────────────────────────
_OTHER_COLORS = [
    "#2196F3",  # Blue
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Grey
    "#3F51B5",  # Indigo
    "#009688",  # Teal
    "#8BC34A",  # Light Green
    "#CDDC39",  # Lime
]

CLASS_COLORS = []
other_iter = iter(_OTHER_COLORS)
for idx in range(NUM_CLASSES):
    if idx == FORGET_IDX:
        CLASS_COLORS.append("#FF0000")   # Bright Neon Red
    else:
        CLASS_COLORS.append(next(other_iter))


# =============================================================================
# 1.  Load dataset & model
# =============================================================================
print("Loading test split ...")
_,_,dataset = get_base_splits()
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Loading model from: {MODEL_PATH}")
model = DcaseMultimodalModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =============================================================================
# 2.  Extract embeddings
# =============================================================================
all_fused, all_vid, all_aud = [], [], []
all_preds_fused, all_preds_vid, all_preds_aud = [], [], []
all_gt_labels = []  # keep ground truth for reference if needed

print("Extracting embeddings ...")
with torch.no_grad():
    for batch in loader:
        video  = batch["video"].to(DEVICE)
        spec   = batch["spectrogram"].to(DEVICE)
        labels = batch["label"]

        out = model(video, spec, return_intermediate=True)

        all_fused.append(out["fused_emb"].cpu().numpy())
        all_vid.append(out["vid_emb"].cpu().numpy())
        all_aud.append(out["aud_emb"].cpu().numpy())

        all_preds_fused.append(out["fusion_logits"].argmax(dim=1).cpu().numpy())
        all_preds_vid.append(out["video_logits"].argmax(dim=1).cpu().numpy())
        all_preds_aud.append(out["audio_logits"].argmax(dim=1).cpu().numpy())

        all_gt_labels.append(labels.numpy())

all_fused  = np.concatenate(all_fused,  axis=0)
all_vid    = np.concatenate(all_vid,    axis=0)
all_aud    = np.concatenate(all_aud,    axis=0)

all_preds_fused = np.concatenate(all_preds_fused, axis=0)
all_preds_vid   = np.concatenate(all_preds_vid,   axis=0)
all_preds_aud   = np.concatenate(all_preds_aud,   axis=0)
all_gt_labels   = np.concatenate(all_gt_labels,   axis=0)

print(f"  Total samples: {len(all_gt_labels)}")


# =============================================================================
# 3.  t-SNE helper
# =============================================================================
def run_tsne(embeddings: np.ndarray, tag: str) -> np.ndarray:
    """Project (N, D) embeddings → (N, 2) using t-SNE."""
    print(f"Running t-SNE on {tag} embeddings {embeddings.shape} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITER,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def save_tsne_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
):
    """Scatter plot — white bg, legend outside to the right."""
    fig, ax = plt.subplots(figsize=(12, 8))   # wider to accommodate outside legend
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw non-forget classes first (background layer)
    for cls_idx in range(NUM_CLASSES):
        if cls_idx == FORGET_IDX:
            continue
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLORS[cls_idx],
            s=12,               # Slightly smaller
            alpha=0.6,          # Slightly more transparent
            linewidths=0,
            label=DCASE_CLASSES[cls_idx],
        )

    # Draw forget class on top (foreground layer)
    mask_forget = labels == FORGET_IDX
    ax.scatter(
        coords[mask_forget, 0],
        coords[mask_forget, 1],
        c=CLASS_COLORS[FORGET_IDX],
        s=20,                   # Moderately larger
        alpha=0.85,             # Almost fully opaque
        linewidths=0.5,
        edgecolor="black",
        label=f"{DCASE_CLASSES[FORGET_IDX]} (forget)",
        zorder=10,              # On top
    )

    ax.set_title(title, color="black", fontsize=14, pad=12)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    # Legend outside the plot, to the right
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        labelcolor="black",
        fontsize=8,
        markerscale=1.5,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])   # leave room on the right for legend
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {save_path}")


# =============================================================================
# 4.  Run & save
# =============================================================================
embeddings_cfg = [
    (all_fused, "fused_emb", "t-SNE — Fused Embedding"),
    (all_vid,   "vid_emb",   "t-SNE — Video Embedding"),
    (all_aud,   "aud_emb",   "t-SNE — Audio Embedding"),
]

for emb, tag, title in embeddings_cfg:
    coords = run_tsne(emb, tag)
    save_tsne_plot(
        coords,
        all_gt_labels,          # ← Use ground truth labels for coloring
        f"{title}\n(coloured by ground truth class)",
        os.path.join(OUTPUT_DIR, f"tsne_{tag}_gt.png"),
    )
