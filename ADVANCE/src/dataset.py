"""
ADVANCE Dataset — Paired Audio-Image Dataset Loader
=====================================================
Each sample pairs:
  - Image  : vision/<class>/<id>.jpg   (RGB image)
  - Audio  : sound/<class>/<id>.wav    → Mel-spectrogram (3-channel image)

Both are processed through ResNet50-compatible transforms.

Directory layout expected:
  ADVANCE_ROOT/
    vision/<class>/<id>.jpg
    sound/<class>/<id>.wav
"""

import os
import hashlib
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not found. Audio loading will fail. Install with: pip install librosa")

from src.labels import LABEL_MAP, NUM_CLASSES

# ─── Paths ────────────────────────────────────────────────────────────────────
ADVANCE_ROOT    = "/home/team2/Unlearning/ADVANCE"
VISION_ROOT     = os.path.join(ADVANCE_ROOT, "data", "vision")
SOUND_ROOT      = os.path.join(ADVANCE_ROOT, "data", "sound")

ANNOTATIONS_DIR = "/home/team2/Unlearning/newDirauth2/ADVANCE_Unlearning/data/annotations"
SPEC_CACHE_DIR  = "/home/team2/Unlearning/newDirauth2/ADVANCE_Unlearning/data/spectrogram_cache"
os.makedirs(SPEC_CACHE_DIR, exist_ok=True)


# ─── Image Transforms (ResNet50 compatible) ───────────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Same transform used for spectrogram (also rendered as RGB image)
SPEC_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Audio → Mel-Spectrogram ───────────────────────────────────────────────────
def wav_to_spectrogram_image(wav_path: str, n_mels: int = 128, sr: int = 22050) -> Image.Image:
    """
    Load a .wav file and convert it to a 3-channel mel-spectrogram PIL image.
    Uses disk caching to avoid reprocessing on every epoch.
    """
    # Build a deterministic cache path from the wav path
    cache_key = hashlib.md5(wav_path.encode()).hexdigest()
    cache_path = os.path.join(SPEC_CACHE_DIR, f"{cache_key}.png")

    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    # Load audio
    y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)

    # Mel-spectrogram (dB scale)
    mel = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=n_mels, fmax=sr_loaded // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalise to [0, 255]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)
    mel_uint8 = (mel_norm * 255).astype(np.uint8)

    # Flip vertically (low freq at bottom) and convert to RGB
    mel_uint8 = np.flipud(mel_uint8)
    img_gray = Image.fromarray(mel_uint8, mode="L")
    img_rgb  = img_gray.convert("RGB")

    # Cache to disk
    img_rgb.save(cache_path)

    return img_rgb


# ─── Dataset ──────────────────────────────────────────────────────────────────
class AdvanceDataset(Dataset):
    """
    Paired Audio-Image dataset for ADVANCE.

    Each sample returns:
        {
            "image"      : Tensor [3, 224, 224]  – vision JPG
            "spectrogram": Tensor [3, 224, 224]  – mel-spectrogram of audio
            "label"      : LongTensor scalar
            "stem"       : str (filename stem, e.g. "00063")
            "class_name" : str (e.g. "airport")
        }

    Args:
        pairs  : list of (vision_path, sound_path, class_name)
    """

    def __init__(self, pairs: list):
        self.pairs = pairs  # list of (img_path, wav_path, class_name)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, wav_path, class_name = self.pairs[idx]

        # ── Vision image ──────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        image = IMAGE_TRANSFORM(image)

        # ── Audio → spectrogram ───────────────────────────────────────────────
        spec_img  = wav_to_spectrogram_image(wav_path)
        spectrogram = SPEC_TRANSFORM(spec_img)

        # ── Label ─────────────────────────────────────────────────────────────
        label = LABEL_MAP[class_name]

        stem = os.path.splitext(os.path.basename(img_path))[0]

        return {
            "image"      : image,
            "spectrogram": spectrogram,
            "label"      : torch.tensor(label, dtype=torch.long),
            "stem"       : stem,
            "class_name" : class_name,
        }


# ─── Pair Discovery ────────────────────────────────────────────────────────────
def discover_pairs(vision_root: str = VISION_ROOT,
                   sound_root: str  = SOUND_ROOT) -> list:
    """
    Walk vision/<class>/ and find matching sound/<class>/<stem>.wav.
    Returns a sorted list of (img_path, wav_path, class_name).
    """
    pairs = []
    for class_name in sorted(os.listdir(vision_root)):
        vis_dir   = os.path.join(vision_root, class_name)
        sound_dir = os.path.join(sound_root,  class_name)

        if not os.path.isdir(vis_dir):
            continue
        if class_name not in LABEL_MAP:
            warnings.warn(f"Class '{class_name}' not in LABEL_MAP, skipping.")
            continue

        for fname in sorted(os.listdir(vis_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            stem    = os.path.splitext(fname)[0]
            wav_name = stem + ".wav"
            wav_path = os.path.join(sound_dir, wav_name)
            if not os.path.exists(wav_path):
                warnings.warn(f"Missing audio for {wav_path}, skipping pair.")
                continue
            img_path = os.path.join(vis_dir, fname)
            pairs.append((img_path, wav_path, class_name))

    return pairs


def get_full_dataset() -> AdvanceDataset:
    pairs = discover_pairs()
    return AdvanceDataset(pairs)


def create_train_val_split(val_ratio: float = 0.2, seed: int = 42):
    """
    Returns (train_dataset, val_dataset).
    """
    from torch.utils.data import random_split
    full = get_full_dataset()
    n    = len(full)
    n_val   = int(n * val_ratio)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return train_ds, val_ds


# ─── Forget / Retain factories ────────────────────────────────────────────────
# "Forget" class: choose one class to unlearn (e.g., "airport")
FORGET_CLASS = "airport"

def ForgetDataset(forget_class: str = FORGET_CLASS) -> AdvanceDataset:
    """Returns all paired samples belonging to the forget class."""
    all_pairs = discover_pairs()
    forget_pairs = [(i, w, c) for i, w, c in all_pairs if c == forget_class]
    return AdvanceDataset(forget_pairs)

def RetainDataset(forget_class: str = FORGET_CLASS) -> AdvanceDataset:
    """Returns all paired samples NOT belonging to the forget class."""
    all_pairs = discover_pairs()
    retain_pairs = [(i, w, c) for i, w, c in all_pairs if c != forget_class]
    return AdvanceDataset(retain_pairs)

from torch.utils.data import random_split

def get_base_splits(val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Returns train, val, test splits on FULL dataset
    """
    full = get_full_dataset()
    n = len(full)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    return train_ds, val_ds, test_ds


def get_forget_splits(forget_class=FORGET_CLASS, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Splits ONLY forget class data
    """
    dataset = ForgetDataset(forget_class)
    n = len(dataset)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    return train_ds, val_ds, test_ds


def get_retain_splits(forget_class=FORGET_CLASS, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Splits ONLY retain data (everything except forget class)
    """
    dataset = RetainDataset(forget_class)
    n = len(dataset)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

    return train_ds, val_ds, test_ds