"""Dataset utilities for corruption benchmarks.

- CIFAR-10-C / CIFAR-100-C support (numpy blobs)
- Tiny-ImageNet-C support (folder-per-class)

Refer to https://github.com/hendrycks/robustness
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _default_transform(image_size: int = 224) -> transforms.Compose:
    """Simple resize + ToTensor transform (no normalization)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


class CIFARCorruptionDataset(Dataset):
    """CIFAR-10-C / CIFAR-100-C loader.

    Each corruption .npy stacks all five severities; we slice the block we need.
    Returned items are dicts with keys: pixel_values (C,H,W tensor), label (int),
    corruption (str), severity (int), index (int), path (str-like id).
    """

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int = 1,
        dataset_name: str = "CIFAR-10-C",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        mmap: bool = True,
    ) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in [1, 5]")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.dataset_name = dataset_name
        self.transform = transform or _default_transform()

        data_dir = self.root / dataset_name
        data_path = data_dir / f"{corruption}.npy"
        labels_path = data_dir / "labels.npy"

        if not data_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {data_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        mmap_mode = "r" if mmap else None
        self.images = np.load(data_path, mmap_mode=mmap_mode)
        self.labels = np.load(labels_path, mmap_mode=mmap_mode)

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("Images and labels have mismatched length")

        # CIFAR-C packs all five severities sequentially
        self.block_size = self.images.shape[0] // 5
        start = (severity - 1) * self.block_size
        end = severity * self.block_size
        self.offset = start
        self.length = end - start

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        real_idx = self.offset + idx
        img_np = self.images[real_idx]

        # CIFAR-C arrays are HWC uint8; convert to PIL for transforms
        img = Image.fromarray(img_np.astype(np.uint8))
        label = int(self.labels[real_idx])

        if self.transform is not None:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "label": label,
            "corruption": self.corruption,
            "severity": self.severity,
            "index": real_idx,
            # acts as a stable identifier if needed in RL logging
            "path": f"{self.dataset_name}/{self.corruption}/{self.severity}/{real_idx}",
        }


class TinyImageNetCorruptionDataset(Dataset):
    """Tiny-ImageNet-C loader.

    Expects layout: root/<corruption>/<severity>/<class_id>/*.JPEG
    Returns dicts with pixel_values, label, path, class_id, corruption, severity.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"}

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int = 1,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in [1, 5]")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform or _default_transform()

        base = self.root / corruption / str(severity)
        if not base.exists():
            raise FileNotFoundError(f"Missing Tiny-ImageNet-C split at: {base}")

        classes = sorted([p.name for p in base.iterdir() if p.is_dir()])
        if not classes:
            raise RuntimeError(f"No class folders found under {base}")

        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.samples: List[Tuple[Path, int]] = []
        for cls in classes:
            for img_path in (base / cls).glob("*"):
                if img_path.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((img_path, self.class_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No images found in {base}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "label": label,
            "path": str(path),
            "class_id": self.idx_to_class[label],
            "corruption": self.corruption,
            "severity": self.severity,
        }


def make_cifar_c_loader(
    root: str | Path,
    corruption: str,
    severity: int = 1,
    dataset_name: str = "CIFAR-10-C",
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> DataLoader:
    dataset = CIFARCorruptionDataset(
        root=root,
        corruption=corruption,
        severity=severity,
        dataset_name=dataset_name,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_tiny_imagenet_c_loader(
    root: str | Path,
    corruption: str,
    severity: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> DataLoader:
    dataset = TinyImageNetCorruptionDataset(
        root=root,
        corruption=corruption,
        severity=severity,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def available_cifar_corruptions(
    root: str | Path, dataset_name: str = "CIFAR-10-C"
) -> List[str]:
    """List corruption names present in a CIFAR-C directory."""
    base = Path(root) / dataset_name
    npy_files = [p.stem for p in base.glob("*.npy") if p.stem != "labels"]
    return sorted(npy_files)
