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
import cv2

import numpy as np
import torch
import random
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
CREMA_ROOT    = "/home/team2/Unlearning/crema-d-mirror"
VISION_ROOT   = os.path.join(CREMA_ROOT, "VideoFlash")
SOUND_ROOT    = os.path.join(CREMA_ROOT, "AudioWAV")

ANNOTATIONS_DIR = os.path.join(CREMA_ROOT, "data", "annotations")
SPEC_CACHE_DIR  = os.path.join(CREMA_ROOT, "data", "spectrogram_cache")
os.makedirs(SPEC_CACHE_DIR, exist_ok=True)


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

# Same transform used for spectrogram (also rendered as RGB image)
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


# ─── Video → Frame ─────────────────────────────────────────────────────────────
def video_to_image(video_path: str, is_train: bool = False) -> Image.Image:
    """
    Open .flv video, extract a frame, and return it as a PIL RGB Image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if is_train and total_frames > 4:
        mid_frame = random.randint(total_frames // 4, 3 * total_frames // 4)
    else:
        mid_frame = max(0, total_frames // 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        # Fallback to a black image if reading fails
        return Image.new("RGB", (224, 224), (0, 0, 0))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class CremaDataset(Dataset):
    """
    Paired Audio-Video dataset for CREMA-D.

    Each sample returns:
        {
            "image"      : Tensor [3, 224, 224]  – video frame
            "spectrogram": Tensor [3, 224, 224]  – mel-spectrogram of audio
            "label"      : LongTensor scalar
            "stem"       : str (filename stem, e.g. "1001_DFA_ANG_XX")
            "class_name" : str (e.g. "ANG")
        }

    Args:
        pairs  : list of (video_path, sound_path, class_name)
    """

    def __init__(self, pairs: list, is_train: bool = False):
        self.pairs = pairs  # list of (video_path, wav_path, class_name)
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_path, wav_path, class_name = self.pairs[idx]

        # ── Video frame ───────────────────────────────────────────────────────
        video = video_to_image(video_path, is_train=self.is_train)
        video = TRAIN_TRANSFORM(video) if self.is_train else VAL_TRANSFORM(video)

        # ── Audio → spectrogram ───────────────────────────────────────────────
        spec_img  = wav_to_spectrogram_image(wav_path)
        spectrogram = SPEC_TRAIN_TRANSFORM(spec_img) if self.is_train else SPEC_VAL_TRANSFORM(spec_img)

        # ── Label ─────────────────────────────────────────────────────────────
        label = LABEL_MAP[class_name]

        stem = os.path.splitext(os.path.basename(video_path))[0]

        return {
            "video"      : video,
            "spectrogram": spectrogram,
            "label"      : torch.tensor(label, dtype=torch.long),
            "stem"       : stem,
            "class_name" : class_name,
        }


# ─── Pair Discovery ────────────────────────────────────────────────────────────
def discover_pairs(vision_root: str = VISION_ROOT,
                   sound_root: str  = SOUND_ROOT) -> list:
    """
    Scan AudioWAV and VideoFlash for matched CREMA-D pairs.
    Returns a sorted list of (video_path, wav_path, class_name).
    """
    pairs = []
    
    if not os.path.isdir(vision_root) or not os.path.isdir(sound_root):
        warnings.warn("CREMA-D directories not found.")
        return pairs
        
    for fname in sorted(os.listdir(vision_root)):
        if not fname.lower().endswith(".flv"):
            continue
            
        stem = os.path.splitext(fname)[0]
        wav_name = stem + ".wav"
        wav_path = os.path.join(sound_root, wav_name)
        
        if not os.path.exists(wav_path):
            warnings.warn(f"Missing audio for {wav_path}, skipping pair.")
            continue
            
        # Extract class name from stem (e.g. 1001_DFA_ANG_XX -> ANG)
        parts = stem.split("_")
        if len(parts) >= 3:
            class_name = parts[2]
            if class_name not in LABEL_MAP:
                warnings.warn(f"Class '{class_name}' not in LABEL_MAP, skipping.")
                continue
            
            video_path = os.path.join(vision_root, fname)
            pairs.append((video_path, wav_path, class_name))

    return pairs


def get_full_dataset() -> CremaDataset:
    pairs = discover_pairs()
    return CremaDataset(pairs)


def get_base_splits(seed: int = 42):
    """
    Returns (train_ds, val_ds, test_ds) using an 80/10/10 split.
    The split is permanent/consistent based on the seed.
    """
    from torch.utils.data import random_split
    import torch
    
    full_pairs = discover_pairs()
    n    = len(full_pairs)
    n_test  = int(n * 0.1)
    n_val   = int(n * 0.1)
    n_train = n - n_val - n_test
    
    gen = torch.Generator().manual_seed(seed)
    splits = random_split(range(n), [n_train, n_val, n_test], generator=gen)
    
    train_pairs = [full_pairs[i] for i in splits[0].indices]
    val_pairs   = [full_pairs[i] for i in splits[1].indices]
    test_pairs  = [full_pairs[i] for i in splits[2].indices]
    
    return (
        CremaDataset(train_pairs, is_train=True),
        CremaDataset(val_pairs, is_train=False),
        CremaDataset(test_pairs, is_train=False)
    )

def filter_dataset_by_class(dataset, target_class: str, keep_target: bool = True):
    from torch.utils.data import Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_ds = dataset.dataset
        subset_indices = dataset.indices
    else:
        base_ds = dataset
        subset_indices = range(len(dataset))
        
    indices = []
    for i in subset_indices:
        _, _, cls_name = base_ds.pairs[i]
        if (cls_name == target_class) == keep_target:
            indices.append(i)
            
    return Subset(base_ds, indices)

# ─── Forget / Retain factories ────────────────────────────────────────────────
# "Forget" class: choose one class to unlearn (e.g., "ANG")
FORGET_CLASS = "HAP"

def get_forget_splits(forget_class: str = FORGET_CLASS, seed: int = 42):
    train_ds, val_ds, test_ds = get_base_splits(seed)
    return (
        filter_dataset_by_class(train_ds, forget_class, keep_target=True),
        filter_dataset_by_class(val_ds, forget_class, keep_target=True),
        filter_dataset_by_class(test_ds, forget_class, keep_target=True)
    )

def get_retain_splits(forget_class: str = FORGET_CLASS, seed: int = 42):
    train_ds, val_ds, test_ds = get_base_splits(seed)
    return (
        filter_dataset_by_class(train_ds, forget_class, keep_target=False),
        filter_dataset_by_class(val_ds, forget_class, keep_target=False),
        filter_dataset_by_class(test_ds, forget_class, keep_target=False)
    )

