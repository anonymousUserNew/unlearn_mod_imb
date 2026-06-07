"""
Generate annotation CSV files for ADVANCE dataset splits.
===========================================================
Creates:
  data/annotations/train.csv   – 80% of full dataset
  data/annotations/val.csv     – 20% of full dataset
  data/annotations/full.csv    – all pairs
  data/annotations/forget.csv  – only the forget class
  data/annotations/retain.csv  – all classes except the forget class

CSV columns: image_path, audio_path, class_name, label_idx

Usage:
    python scripts/make_annotations.py
    python scripts/make_annotations.py --forget_class beach
"""

import os
import sys
import csv
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import discover_pairs, FORGET_CLASS
from src.labels  import LABEL_MAP

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR     = os.path.join(BASE_DIR, "data", "annotations")
os.makedirs(ANN_DIR, exist_ok=True)


def write_csv(path: str, pairs: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "audio_path", "class_name", "label_idx"])
        for img_p, wav_p, cls in pairs:
            writer.writerow([img_p, wav_p, cls, LABEL_MAP[cls]])
    print(f"  Wrote {len(pairs):>5} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_class", default=FORGET_CLASS,
                        help=f"Class to forget (default: {FORGET_CLASS})")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_pairs = discover_pairs()
    print(f"Total discovered pairs: {len(all_pairs)}")

    # Shuffle
    shuffled = all_pairs[:]
    random.shuffle(shuffled)

    # Full
    write_csv(os.path.join(ANN_DIR, "full.csv"), all_pairs)

    # Train / Val split
    n_val   = int(len(shuffled) * args.val_ratio)
    val_p   = shuffled[:n_val]
    train_p = shuffled[n_val:]
    write_csv(os.path.join(ANN_DIR, "train.csv"), train_p)
    write_csv(os.path.join(ANN_DIR, "val.csv"),   val_p)

    # Forget / Retain
    forget_p = [(i, w, c) for i, w, c in all_pairs if c == args.forget_class]
    retain_p = [(i, w, c) for i, w, c in all_pairs if c != args.forget_class]
    write_csv(os.path.join(ANN_DIR, "forget.csv"), forget_p)
    write_csv(os.path.join(ANN_DIR, "retain.csv"), retain_p)

    print("\nAnnotations saved to:", ANN_DIR)
    print(f"  Forget class : '{args.forget_class}' ({len(forget_p)} samples)")
    print(f"  Retain       : {len(retain_p)} samples")


if __name__ == "__main__":
    main()
