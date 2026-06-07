"""
Sanity check: verify DCASE dataset loads correctly.

Usage:
    cd /home/team2/Unlearning/Dcase
    python scripts/verify_dataset.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.dataset import discover_pairs, get_base_splits, get_forget_splits, get_retain_splits, FORGET_CLASS
from src.labels  import LABEL_MAP, NUM_CLASSES, DCASE_CLASSES

print("=" * 60)
print("  DCASE Dataset Sanity Check")
print("=" * 60)

# ── 1. Pair discovery ─────────────────────────────────────────
train_p = discover_pairs("train")
val_p   = discover_pairs("val")
test_p  = discover_pairs("test")

print(f"\n[1] Pair discovery:")
print(f"    Train : {len(train_p):>5} pairs")
print(f"    Val   : {len(val_p):>5} pairs")
print(f"    Test  : {len(test_p):>5} pairs")
print(f"    Total : {len(train_p)+len(val_p)+len(test_p):>5} pairs")

readme_expected = {"train": 8508, "val": 1249, "test": 2534}
for split, expected in readme_expected.items():
    found = len(discover_pairs(split))
    status = "✅" if found == expected else f"⚠️  expected {expected}"
    print(f"    {split}: {found} {status}")

# ── 2. Label check ────────────────────────────────────────────
print(f"\n[2] Label map ({NUM_CLASSES} classes):")
for cls, idx in LABEL_MAP.items():
    print(f"    {idx:>2}: {cls}")

# ── 3. Single sample loading ──────────────────────────────────
print(f"\n[3] Loading one sample from train ...")
train_ds, val_ds, test_ds = get_base_splits()
sample = train_ds[0]
print(f"    video shape      : {sample['video'].shape}  dtype={sample['video'].dtype}")
print(f"    spectrogram shape: {sample['spectrogram'].shape}  dtype={sample['spectrogram'].dtype}")
print(f"    label            : {sample['label'].item()} ({sample['class_name']})")
print(f"    stem             : {sample['stem']}")

assert sample["video"].shape == torch.Size([3, 224, 224]), "Video shape mismatch!"
assert sample["spectrogram"].shape == torch.Size([3, 224, 224]), "Spectrogram shape mismatch!"
assert 0 <= sample["label"].item() < NUM_CLASSES, "Label out of range!"

# ── 4. DataLoader batch ───────────────────────────────────────
print(f"\n[4] Running one DataLoader batch ...")
loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
batch  = next(iter(loader))
print(f"    video batch shape      : {batch['video'].shape}")
print(f"    spectrogram batch shape: {batch['spectrogram'].shape}")
print(f"    labels                 : {batch['label'].tolist()}")

# ── 5. Forget / Retain splits ─────────────────────────────────
print(f"\n[5] Forget/Retain splits (forget_class='{FORGET_CLASS}') ...")
ft, fv, fte = get_forget_splits()
rt, rv, rte = get_retain_splits()
print(f"    Forget → train: {len(ft)}  val: {len(fv)}  test: {len(fte)}")
print(f"    Retain → train: {len(rt)}  val: {len(rv)}  test: {len(rte)}")

print("\n✅  All checks passed! Dataset is ready.\n")
