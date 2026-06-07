"""
Generate annotation CSV files for DCASE dataset splits.
=========================================================
Creates:
  data/annotations/train.csv   – pre-split train set
  data/annotations/val.csv     – pre-split val set
  data/annotations/test.csv    – pre-split test set
  data/annotations/full.csv    – all pairs (train+val+test)
  data/annotations/forget.csv  – only the forget class
  data/annotations/retain.csv  – all classes except the forget class

CSV columns: video_path, audio_path, class_name, label_idx

Usage:
    python scripts/make_annotations.py
    python scripts/make_annotations.py --forget_class metro
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import discover_pairs, FORGET_CLASS
from src.labels  import LABEL_MAP

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR     = os.path.join(BASE_DIR, "data", "annotations")
os.makedirs(ANN_DIR, exist_ok=True)


def write_csv(path: str, pairs: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "audio_path", "class_name", "label_idx"])
        for vid_p, h5_p, cls in pairs:
            writer.writerow([vid_p, h5_p, cls, LABEL_MAP[cls]])
    print(f"  Wrote {len(pairs):>5} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_class", default=FORGET_CLASS,
                        help=f"Class to forget (default: {FORGET_CLASS})")
    args = parser.parse_args()

    train_pairs = discover_pairs("train")
    val_pairs   = discover_pairs("val")
    test_pairs  = discover_pairs("test")
    all_pairs   = train_pairs + val_pairs + test_pairs

    print(f"Discovered: Train={len(train_pairs)} | Val={len(val_pairs)} | Test={len(test_pairs)}"
          f" | Total={len(all_pairs)}")

    write_csv(os.path.join(ANN_DIR, "full.csv"),  all_pairs)
    write_csv(os.path.join(ANN_DIR, "train.csv"), train_pairs)
    write_csv(os.path.join(ANN_DIR, "val.csv"),   val_pairs)
    write_csv(os.path.join(ANN_DIR, "test.csv"),  test_pairs)

    forget_p = [(v, a, c) for v, a, c in all_pairs if c == args.forget_class]
    retain_p = [(v, a, c) for v, a, c in all_pairs if c != args.forget_class]
    write_csv(os.path.join(ANN_DIR, "forget.csv"), forget_p)
    write_csv(os.path.join(ANN_DIR, "retain.csv"), retain_p)

    print("\nAnnotations saved to:", ANN_DIR)
    print(f"  Forget class : '{args.forget_class}' ({len(forget_p)} samples)")
    print(f"  Retain       : {len(retain_p)} samples")


if __name__ == "__main__":
    main()
