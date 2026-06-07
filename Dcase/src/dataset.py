"""
DCASE Acoustic Scene Dataset — Paired Audio-Video Loader
=========================================================
Each sample pairs:
  - Video : data/archive/{split}/{split}/video/<class>/<stem>.pt   → uint8 [16,3,224,224]
  - Audio : data/archive/{split}/{split}/audio/<class>/<stem>.h5   → mel-spectrogram

The dataset is pre-split:
  train → data/archive/train/train/
  val   → data/archive/val/val/
  test  → data/archive/test/test/

Directory layout:
  data/archive/
    train/train/audio/<class>/<stem>.h5
    train/train/video/<class>/<stem>.pt
    val/val/audio/<class>/<stem>.h5
    val/val/video/<class>/<stem>.pt
    test/test/audio/<class>/<stem>.h5
    test/test/video/<class>/<stem>.pt
"""

import os
import warnings
import numpy as np
import torch
import random
from torch.utils.data import Dataset, Subset
from PIL import Image
from torchvision import transforms

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not found. Audio loading will fail. Install with: pip install h5py")

from src.labels import LABEL_MAP, NUM_CLASSES

# ─── Paths ────────────────────────────────────────────────────────────────────
DCASE_ROOT = "/home/team2/Unlearning/Dcase/data/archive"

# Pre-defined split roots
SPLIT_ROOTS = {
    "train": os.path.join(DCASE_ROOT, "train", "train"),
    "val":   os.path.join(DCASE_ROOT, "val",   "val"),
    "test":  os.path.join(DCASE_ROOT, "test",  "test"),
}

# ─── Image Transforms (ResNet50 compatible) ───────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SPEC_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
])

SPEC_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Audio → Mel-Spectrogram image ────────────────────────────────────────────
def load_audio_h5(h5_path: str) -> Image.Image:
    """
    Load a DCASE .h5 audio file and return a 3-channel PIL image.

    The h5 file is expected to contain a mel-spectrogram or raw waveform.
    Common DCASE keys tried in order: 'mel_spectrogram', 'data', 'audio'.
    If none found, uses the first dataset key.
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py is required: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        # Try common DCASE key names in order
        data = None
        for key in ("mel_spectrogram", "mel", "data", "audio", "features"):
            if key in f:
                data = f[key][()]
                break
        if data is None:
            # Fall back to first dataset
            for key in f.keys():
                try:
                    data = f[key][()]
                    break
                except Exception:
                    continue

    if data is None:
        warnings.warn(f"Could not load any data from {h5_path}. Returning blank image.")
        return Image.new("RGB", (224, 224), (0, 0, 0))

    # Handle possible shapes
    # Could be (time_frames, mel_bins), (mel_bins, time_frames), (1, mel_bins, time_frames), etc.
    data = np.squeeze(data)  # remove singleton dims

    if data.ndim == 1:
        # raw waveform — reshape into 2D for display
        n = int(np.sqrt(len(data))) + 1
        pad = n * n - len(data)
        data = np.pad(data, (0, pad)).reshape(n, n)
    elif data.ndim == 3:
        # Already a spectrogram image (C, H, W) — use channel 0
        data = data[0]

    # Now data is 2D: normalize to [0, 255]
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data_norm = (data - d_min) / (d_max - d_min)
    else:
        data_norm = np.zeros_like(data)

    img_uint8 = (data_norm * 255).astype(np.uint8)
    img_uint8 = np.flipud(img_uint8)           # frequency axis: low at bottom

    img = Image.fromarray(img_uint8, mode="L").convert("RGB")
    return img


# ─── Video → Single frame ─────────────────────────────────────────────────────
def load_video_pt(pt_path: str, is_train: bool = False) -> torch.Tensor:
    """
    Load a DCASE .pt video file (uint8 tensor [16, 3, 224, 224]).
    Selects one frame (random middle half during training, middle frame for val/test).
    Converts uint8 → float32 / 255 and applies ImageNet normalisation.

    Returns: Tensor [3, 224, 224] (float32, normalised)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    video = torch.load(pt_path, weights_only=True)    # [16, 3, 224, 224] uint8
    n_frames = video.shape[0]

    if is_train and n_frames > 4:
        mid_lo = n_frames // 4
        mid_hi = 3 * n_frames // 4
        idx = random.randint(mid_lo, mid_hi)
    else:
        idx = n_frames // 2

    frame = video[idx].to(torch.float32) / 255.0     # [3, 224, 224] float in [0,1]
    frame = normalize(frame)
    return frame


# ─── Dataset ──────────────────────────────────────────────────────────────────
class DcaseDataset(Dataset):
    """
    Paired Audio-Video dataset for DCASE acoustic scenes.

    Each sample returns:
        {
            "video"      : Tensor [3, 224, 224]  – video frame (normalised)
            "spectrogram": Tensor [3, 224, 224]  – mel-spectrogram from .h5
            "label"      : LongTensor scalar
            "stem"       : str (filename stem)
            "class_name" : str (e.g. "airport")
        }

    Args:
        pairs    : list of (video_path, h5_path, class_name)
        is_train : bool — enables augmentations
    """

    def __init__(self, pairs: list, is_train: bool = False):
        self.pairs    = pairs
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_path, h5_path, class_name = self.pairs[idx]

        # ── Video frame ───────────────────────────────────────────────────────
        video = load_video_pt(video_path, is_train=self.is_train)
        if self.is_train:
            # Apply random crop + flip on top of the loaded frame
            # Re-convert to PIL for torchvision transforms, then back
            # (We still want augmentations like RandomCrop after single-frame pick)
            frame_pil = transforms.ToPILImage()(
                (video * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) +
                 torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)).clamp(0, 1)
            )
            video = TRAIN_TRANSFORM(frame_pil)
        # else: video is already [3,224,224] normalised — no further transform needed

        # ── Audio → spectrogram ───────────────────────────────────────────────
        spec_img    = load_audio_h5(h5_path)
        spectrogram = SPEC_TRAIN_TRANSFORM(spec_img) if self.is_train else SPEC_VAL_TRANSFORM(spec_img)

        # ── Label ─────────────────────────────────────────────────────────────
        label = LABEL_MAP[class_name]
        stem  = os.path.splitext(os.path.basename(h5_path))[0]

        return {
            "video"      : video,
            "spectrogram": spectrogram,
            "label"      : torch.tensor(label, dtype=torch.long),
            "stem"       : stem,
            "class_name" : class_name,
        }


# ─── Pair Discovery ────────────────────────────────────────────────────────────
def discover_pairs_for_split(split_root: str) -> list:
    """
    Discover paired (video_path, audio_h5_path, class_name) tuples for one split.

    Directory structure expected:
        split_root/audio/<class>/<stem>.h5
        split_root/video/<class>/<stem>.pt

    Returns a sorted list of (video_pt_path, audio_h5_path, class_name).
    """
    audio_root = os.path.join(split_root, "audio")
    video_root = os.path.join(split_root, "video")

    pairs = []

    if not os.path.isdir(audio_root) or not os.path.isdir(video_root):
        warnings.warn(f"audio or video dir missing under {split_root}")
        return pairs

    for class_name in sorted(os.listdir(audio_root)):
        if class_name not in LABEL_MAP:
            continue  # skip unknown classes

        audio_class_dir = os.path.join(audio_root, class_name)
        video_class_dir = os.path.join(video_root, class_name)

        if not os.path.isdir(audio_class_dir):
            continue

        for audio_fname in sorted(os.listdir(audio_class_dir)):
            if not audio_fname.lower().endswith(".h5"):
                continue

            stem     = os.path.splitext(audio_fname)[0]
            h5_path  = os.path.join(audio_class_dir, audio_fname)
            pt_path  = os.path.join(video_class_dir, stem + ".pt")

            if not os.path.exists(pt_path):
                warnings.warn(f"Missing video .pt for {stem}, skipping.")
                continue

            pairs.append((pt_path, h5_path, class_name))

    return sorted(pairs, key=lambda x: x[1])


def discover_pairs(split: str = "train") -> list:
    """
    Convenience wrapper. split: 'train' | 'val' | 'test'.
    """
    if split not in SPLIT_ROOTS:
        raise ValueError(f"split must be one of {list(SPLIT_ROOTS.keys())}")
    return discover_pairs_for_split(SPLIT_ROOTS[split])


# ─── Pre-split dataset builders ───────────────────────────────────────────────
def get_base_splits():
    """
    Returns (train_ds, val_ds, test_ds) using the pre-defined DCASE splits.
    """
    train_pairs = discover_pairs("train")
    val_pairs   = discover_pairs("val")
    test_pairs  = discover_pairs("test")
    return (
        DcaseDataset(train_pairs, is_train=True),
        DcaseDataset(val_pairs,   is_train=False),
        DcaseDataset(test_pairs,  is_train=False),
    )


def get_full_dataset() -> DcaseDataset:
    """Returns all pairs from all splits combined (train+val+test)."""
    all_pairs = (
        discover_pairs("train") +
        discover_pairs("val") +
        discover_pairs("test")
    )
    return DcaseDataset(all_pairs, is_train=False)


# ─── Forget / Retain filtering ────────────────────────────────────────────────
FORGET_CLASS = "bus"   # default class to unlearn


def filter_dataset_by_class(dataset, target_class: str, keep_target: bool = True):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_ds        = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_ds        = dataset
        subset_indices = range(len(dataset))

    indices = [
        i for i in subset_indices
        if (base_ds.pairs[i][2] == target_class) == keep_target
    ]
    return Subset(base_ds, indices)


def get_forget_splits(forget_class: str = FORGET_CLASS, seed: int = 42):
    """Returns (forget_train, forget_val, forget_test) datasets."""
    train_ds, val_ds, test_ds = get_base_splits()
    return (
        filter_dataset_by_class(train_ds, forget_class, keep_target=True),
        filter_dataset_by_class(val_ds,   forget_class, keep_target=True),
        filter_dataset_by_class(test_ds,  forget_class, keep_target=True),
    )


def get_retain_splits(forget_class: str = FORGET_CLASS, seed: int = 42):
    """Returns (retain_train, retain_val, retain_test) datasets (all classes except forget)."""
    train_ds, val_ds, test_ds = get_base_splits()
    return (
        filter_dataset_by_class(train_ds, forget_class, keep_target=False),
        filter_dataset_by_class(val_ds,   forget_class, keep_target=False),
        filter_dataset_by_class(test_ds,  forget_class, keep_target=False),
    )
