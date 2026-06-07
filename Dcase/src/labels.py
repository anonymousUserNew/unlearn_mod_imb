"""
DCASE Acoustic Scene Classification – Label Definitions
10 scene classes (alphabetically sorted, matching directory names).
"""

# -------------------------
# Class names (sorted alphabetically — matches subdirectory names)
# -------------------------
DCASE_CLASSES = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]

NUM_CLASSES = len(DCASE_CLASSES)  # 10

# -------------------------
# Label <-> Index mappings
# -------------------------
LABEL_MAP    = {cls: idx for idx, cls in enumerate(DCASE_CLASSES)}
IDX_TO_LABEL = {idx: cls for cls, idx in LABEL_MAP.items()}


def label_to_idx(label: str) -> int:
    return LABEL_MAP[label]


def idx_to_label(idx: int) -> str:
    return IDX_TO_LABEL[idx]
