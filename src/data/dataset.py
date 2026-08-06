"""
src/data/dataset.py
===================
PyTorch Dataset class for the preprocessed Cats vs Dogs dataset.

Expects the processed directory structure:
    data/processed/{split}/{class}/image.jpg
    e.g. data/processed/train/cats/cats_00001.jpg
         data/processed/train/dogs/dogs_00001.jpg

Usage:
    from src.data.dataset import CatsDogsDataset, get_transforms
    dataset = CatsDogsDataset(root_dir="data/processed/train",
                              transform=get_transforms("train"))
"""

from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── Label map ──────────────────────────────────────────────────────────────────
CLASS_TO_IDX = {"cats": 0, "dogs": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return torchvision transforms for a given split.
    - 'train' : augmentation + normalize
    - 'val'   : normalize only
    - 'test'  : normalize only
    """
    # ImageNet normalization (widely used for CNNs trained from scratch too)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val or test — no augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class CatsDogsDataset(Dataset):
    """
    Custom PyTorch Dataset for preprocessed Cats vs Dogs images.

    Args:
        root_dir  (str | Path): Path to split folder,
                                e.g. 'data/processed/train'
        transform (callable) : Optional transform applied to each image.
                                Use get_transforms(split) for standard transforms.
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        # Collect all (image_path, label) pairs
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}. "
                    f"Run preprocess.py first."
                )
            images = sorted(class_dir.glob("*.jpg"))
            for img_path in images:
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                f"Run preprocess.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image as RGB PIL Image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Return human-readable class name for a label index."""
        return IDX_TO_CLASS.get(idx, "unknown")

    def __repr__(self) -> str:
        n_cats = sum(1 for _, lbl in self.samples if lbl == 0)
        n_dogs = sum(1 for _, lbl in self.samples if lbl == 1)
        return (
            f"CatsDogsDataset(root='{self.root_dir}', "
            f"total={len(self.samples)}, cats={n_cats}, dogs={n_dogs})"
        )


# ── DataLoader factory ──────────────────────────────────────────────────────────

def get_dataloader(split: str,
                   processed_root: str = "data/processed",
                   batch_size: int = 32,
                   num_workers: int = 2,
                   shuffle: Optional[bool] = None) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Args:
        split          : 'train', 'val', or 'test'
        processed_root : root of processed data directory
        batch_size     : number of images per batch
        num_workers    : parallel data loading workers
        shuffle        : defaults to True for train, False for val/test
    """
    root_dir  = Path(processed_root) / split
    transform = get_transforms(split)
    dataset   = CatsDogsDataset(root_dir=root_dir, transform=transform)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,   # set True if using GPU
    )
    return loader


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        try:
            ds = CatsDogsDataset(
                root_dir=f"data/processed/{split}",
                transform=get_transforms(split)
            )
            print(ds)
            img, lbl = ds[0]
            print(f"  Sample tensor shape : {img.shape}")
            print(f"  Sample label        : {lbl} ({ds.get_class_name(lbl)})")
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
        print()
