"""
src/data/preprocess.py
======================
Preprocessing pipeline for the Cats vs Dogs dataset.

- Source  : data/raw/PetImages/Cat/ and data/raw/PetImages/Dog/
- Output  : data/processed/{train,val,test}/{cats,dogs}/
- Split   : 80% train / 10% val / 10% test
- Resize  : 224x224 RGB (standard CNN input)
- Skips corrupted/unreadable images automatically

Usage:
    PYTHONPATH=. python src/data/preprocess.py
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_ROOT      = Path("data/raw/PetImages")
PROCESSED_ROOT = Path("data/processed")
IMAGE_SIZE    = (224, 224)
TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
TEST_RATIO    = 0.10
RANDOM_SEED   = 42

CLASS_MAP = {
    "Cat": "cats",
    "Dog": "dogs",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_valid_image(path: Path) -> bool:
    """Return True if the file can be opened as an RGB image."""
    try:
        with Image.open(path) as img:
            img.verify()          # catches truncated files
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            img.convert("RGB")   # catches palette/greyscale edge-cases
        return True
    except Exception:
        return False


def resize_and_save(src: Path, dst: Path) -> None:
    """Open src image, resize to IMAGE_SIZE, save as RGB JPEG to dst."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)


def split_files(files: list, seed: int = RANDOM_SEED):
    """Randomly split a list into (train, val, test) according to ratios."""
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]
    return train, val, test


# ── Main ───────────────────────────────────────────────────────────────────────

def preprocess():
    print("=" * 60)
    print("  Cats vs Dogs — Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Source      : {RAW_ROOT}")
    print(f"  Destination : {PROCESSED_ROOT}")
    print(f"  Image size  : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print(f"  Split       : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print("=" * 60)

    total_skipped = 0
    summary = {}

    for raw_class, out_class in CLASS_MAP.items():
        src_dir = RAW_ROOT / raw_class
        if not src_dir.exists():
            raise FileNotFoundError(f"Expected directory not found: {src_dir}")

        # ── Collect valid image paths ──────────────────────────────────────
        all_files = sorted(src_dir.glob("*.jpg"))
        print(f"\n[{raw_class}] Found {len(all_files)} total files. Validating...")

        valid_files = []
        skipped     = 0
        for f in all_files:
            if is_valid_image(f):
                valid_files.append(f)
            else:
                skipped += 1

        total_skipped += skipped
        print(f"[{raw_class}] Valid: {len(valid_files)}  |  Skipped (corrupt): {skipped}")

        # ── Split ─────────────────────────────────────────────────────────
        train_files, val_files, test_files = split_files(valid_files)
        print(f"[{raw_class}] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

        splits = {
            "train": train_files,
            "val":   val_files,
            "test":  test_files,
        }
        summary[out_class] = {
            "train": len(train_files),
            "val":   len(val_files),
            "test":  len(test_files),
        }

        # ── Resize & save ─────────────────────────────────────────────────
        for split_name, files in splits.items():
            out_dir = PROCESSED_ROOT / split_name / out_class
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, src_path in enumerate(files):
                dst_path = out_dir / f"{out_class}_{i:05d}.jpg"
                resize_and_save(src_path, dst_path)

            print(f"  -> Saved {len(files)} images to {out_dir}")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    total_train = sum(v["train"] for v in summary.values())
    total_val   = sum(v["val"]   for v in summary.values())
    total_test  = sum(v["test"]  for v in summary.values())

    print(f"  Train  : {total_train} images  ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  Val    : {total_val}  images  ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  Test   : {total_test}  images  ({total_test/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  Total  : {total_train+total_val+total_test} images processed")
    print(f"  Skipped: {total_skipped} corrupt images")
    print("=" * 60)
    print("\nOutput structure:")
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            p = PROCESSED_ROOT / split / cls
            count = len(list(p.glob("*.jpg"))) if p.exists() else 0
            print(f"  {p}  ->  {count} images")
    print("\nDone! Run 'dvc add data/processed/' next.")


if __name__ == "__main__":
    preprocess()
