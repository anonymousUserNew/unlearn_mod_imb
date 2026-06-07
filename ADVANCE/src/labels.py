
"""
ADVANCE Dataset Label Definitions
13 scene classes: audio-image paired dataset.
"""

# -------------------------
# Class names (sorted)
# -------------------------
ADVANCE_CLASSES = [
    "airport",
    "beach",
    "bridge",
    "farmland",
    "forest",
    "grassland",
    "harbour",
    "lake",
    "orchard",
    "residential",
    "sparse shrub land",
    "sports land",
    "train station",
]

NUM_CLASSES = len(ADVANCE_CLASSES)  # 13

# -------------------------
# Label <-> Index mappings
# -------------------------
LABEL_MAP = {cls: idx for idx, cls in enumerate(ADVANCE_CLASSES)}
IDX_TO_LABEL = {idx: cls for cls, idx in LABEL_MAP.items()}


def label_to_idx(label: str) -> int:
    return LABEL_MAP[label]


def idx_to_label(idx: int) -> str:
    return IDX_TO_LABEL[idx]
