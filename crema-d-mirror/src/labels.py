"""
ADVANCE Dataset Label Definitions
13 scene classes: audio-image paired dataset.
"""

# -------------------------
# Class names (sorted)
# -------------------------
CREMA_CLASSES = [
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD",
]

NUM_CLASSES = len(CREMA_CLASSES)  # 6

# -------------------------
# Label <-> Index mappings
# -------------------------
LABEL_MAP = {cls: idx for idx, cls in enumerate(CREMA_CLASSES)}
IDX_TO_LABEL = {idx: cls for cls, idx in LABEL_MAP.items()}


def label_to_idx(label: str) -> int:
    return LABEL_MAP[label]


def idx_to_label(idx: int) -> str:
    return IDX_TO_LABEL[idx]
