"""
Mainly stores utils for working with the CIFAR-C dataset.

However, some generic utils -- such as for building reward functions and dataset classes, remains universal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from tqdm import tqdm

from rl.reward import CLIPReward
from data.corruption_datasets import (
    CIFARCorruptionDataset,
    TinyImageNetCorruptionDataset,
)
from data.COD_dataset import build_COD_torch_dataset

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _load_tiny_imagenet_wnid_to_label(clean_root: Path) -> Dict[str, str]:
    """Load Tiny-ImageNet wnid -> human-readable label mapping.

    Expects the standard Tiny-ImageNet-200 layout with ``wnids.txt`` and
    ``words.txt``. We primarily rely on ``words.txt`` which maps
    ``wnid -> comma separated synonyms`` and take the first synonym as a
    concise class name. If files are missing, returns an empty dict and
    callers should gracefully fall back to using the raw WNIDs.
    """

    wnids_path = clean_root / "wnids.txt"
    words_path = clean_root / "words.txt"

    mapping: Dict[str, str] = {}

    if not words_path.exists():
        return mapping

    try:
        with words_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Lines are typically "wnid<TAB>syn1, syn2, ..." or
                # "wnid syn1, syn2, ...". Handle both.
                if "\t" in line:
                    wnid, rest = line.split("\t", 1)
                else:
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    wnid, rest = parts

                human = rest.split(",")[0].strip()
                mapping[wnid] = human
    except Exception:
        # On any unexpected format issue, just return whatever we parsed
        # (possibly empty) and let callers fall back to wnids.
        return mapping

    return mapping


def _load_imagenet_wnid_to_label(clean_root: Path) -> Dict[str, str]:
    """Load ImageNet-1K wnid -> human-readable label mapping.

    Expects ``LOC_synset_mapping.txt`` in the given directory, with lines of
    the form ``wnid<space>syn1, syn2, ...``. We take the first synonym as a
    concise class name. If the file is missing or malformed, returns an empty
    dict and callers should fall back to using raw WNIDs.
    """

    mapping: Dict[str, str] = {}
    mapping_path = clean_root / "LOC_synset_mapping.txt"

    if not mapping_path.exists():
        return mapping

    try:
        with mapping_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                wnid, rest = parts

                human = rest.split(",")[0].strip()
                mapping[wnid] = human
    except Exception:
        return mapping

    return mapping


def build_sd_transform(image_size: int, augment: bool = False):
    """
    Builds the image transform for Stable Diffusion.
    Matches the COD dataset transform if augment is True.
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])
    else:
        # Keep images in [0, 1] for SD; classifier reward will normalize internally.
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
            ]
        )


def resolve_label_str_fn(dataset_name: str, tiny_dataset=None):
    if dataset_name == "CIFAR-10-C":
        return lambda idx: CIFAR10_LABELS[idx]
    if dataset_name == "CIFAR-100-C":
        return lambda idx: str(idx)
    if dataset_name == "Tiny-ImageNet-C" and tiny_dataset is not None:
        return lambda idx: tiny_dataset.idx_to_class[idx]
    return lambda idx: str(idx)


def build_data_and_reward(accelerator: Accelerator, args: Any):
    device = accelerator.device
    sd_transform = build_sd_transform(args.image_size, augment=True) 

    if args.dataset == "cod":
        # COD dataset builds its own transform internally in build_COD_torch_dataset
        dataset = build_COD_torch_dataset("train", image_size=args.image_size)
        label2str = dataset.label2str

        reward_fn = CLIPReward(
            class_names=dataset.all_classes,
            device=device,
            reward_variant=args.reward_variant,
            model_name=args.clip_variant,
            lpips_weight=args.lpips_weight,
        )
    elif args.dataset in ("cifar10-c", "cifar100-c"):
        dataset_name = "CIFAR-10-C" if args.dataset == "cifar10-c" else "CIFAR-100-C"
        if args.corruption is None:
            raise ValueError("--corruption is required for CIFAR-C datasets")

        dataset = CIFARCorruptionDataset(
            root=args.data_root,
            corruption=args.corruption,
            severity=args.severity,
            dataset_name=dataset_name,
            transform=sd_transform,
        )
        if args.dataset == "cifar10-c":
            class_names = CIFAR10_LABELS

            def label2str(idx: int) -> str:
                return CIFAR10_LABELS[idx]

        else:
            class_names = [f"class {i}" for i in range(100)]

            def label2str(idx: int) -> str:  # type: ignore[redefined-outer-name]
                return f"class {idx}"

        reward_fn = CLIPReward(
            class_names=class_names,
            device=device,
            reward_variant=args.reward_variant,
            model_name=args.clip_variant,
            lpips_weight=args.lpips_weight,
        )
    elif args.dataset == "tiny-imagenet-c":
        if args.corruption is None:
            raise ValueError("--corruption is required for Tiny-ImageNet-C")

        tiny_root = Path(args.data_root) / "Tiny-ImageNet-C"
        dataset = TinyImageNetCorruptionDataset(
            root=tiny_root,
            corruption=args.corruption,
            severity=args.severity,
            transform=sd_transform,
        )

        # Map Tiny-ImageNet WNIDs to human-readable labels using the
        # clean tiny-imagenet-200 metadata if available.
        clean_root = Path(args.data_root) / "tiny-imagenet-200"
        wnid_to_label = _load_tiny_imagenet_wnid_to_label(clean_root)

        def label2str(idx: int) -> str:
            wnid = dataset.idx_to_class[idx]
            return wnid_to_label.get(wnid, wnid)

        class_names = [label2str(i) for i in range(len(dataset.idx_to_class))]

        reward_fn = CLIPReward(
            class_names=class_names,
            device=device,
            reward_variant=args.reward_variant,
            model_name=args.clip_variant,
            lpips_weight=args.lpips_weight,
        )
    elif args.dataset == "imagenet-c":
        if args.corruption is None:
            raise ValueError("--corruption is required for ImageNet-C")

        imagenet_root = Path(args.data_root) / "ImageNet-C"
        dataset = TinyImageNetCorruptionDataset(
            root=imagenet_root,
            corruption=args.corruption,
            severity=args.severity,
            transform=sd_transform,
        )

        clean_root = Path(args.data_root) / "ImageNet-1000"
        wnid_to_label = _load_imagenet_wnid_to_label(clean_root)

        def label2str(idx: int) -> str:  # type: ignore[redefined-outer-name]
            wnid = dataset.idx_to_class[idx]
            return wnid_to_label.get(wnid, wnid)

        class_names = [label2str(i) for i in range(len(dataset.idx_to_class))]

        reward_fn = CLIPReward(
            class_names=class_names,
            device=device,
            reward_variant=args.reward_variant,
            model_name=args.clip_variant,
            lpips_weight=args.lpips_weight,
        )
    else:
        raise ValueError(f"Unknown dataset choice: {args.dataset}")

    train_dataset = dataset
    if args.overfit_dset_size > 0:
        train_dataset = Subset(dataset, torch.arange(args.overfit_dset_size))
        if accelerator.is_main_process:
            print(f"Using overfit dataset of size: {args.overfit_dset_size}")
    else:
        if accelerator.is_main_process:
            print("Using full training dataset.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.loader_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    train_loader = accelerator.prepare(train_loader)

    return dataset, label2str, reward_fn, train_loader


def run_classifier_eval(args: Any, device: torch.device) -> None:
    """Pure classifier evaluation on the chosen corruption dataset.

    Uses the same dataset/transform construction as RL mode, but bypasses
    diffusion and reports CLIP-based classification loss/accuracy instead of
    ResNet.
    """
    sd_transform = build_sd_transform(args.image_size, augment=False)

    # Build dataset and CLIP text prompts
    if args.dataset in ("cifar10-c", "cifar100-c"):
        dataset_name = "CIFAR-10-C" if args.dataset == "cifar10-c" else "CIFAR-100-C"
        if args.corruption is None:
            raise ValueError("--corruption is required for CIFAR-C datasets")
        dataset = CIFARCorruptionDataset(
            root=args.data_root,
            corruption=args.corruption,
            severity=args.severity,
            dataset_name=dataset_name,
            transform=sd_transform,
        )
        if args.dataset == "cifar10-c":
            class_names = CIFAR10_LABELS
        else:
            class_names = [f"class {i}" for i in range(100)]
    elif args.dataset == "tiny-imagenet-c":
        if args.corruption is None:
            raise ValueError("--corruption is required for Tiny-ImageNet-C")
        tiny_root = Path(args.data_root) / "Tiny-ImageNet-C"
        dataset = TinyImageNetCorruptionDataset(
            root=tiny_root,
            corruption=args.corruption,
            severity=args.severity,
            transform=sd_transform,
        )
        # Human-readable Tiny-ImageNet labels, falling back to WNIDs
        clean_root = Path(args.data_root) / "tiny-imagenet-200"
        wnid_to_label = _load_tiny_imagenet_wnid_to_label(clean_root)
        class_names = []
        for i in range(len(dataset.idx_to_class)):
            wnid = dataset.idx_to_class[i]
            class_names.append(wnid_to_label.get(wnid, wnid))
    elif args.dataset == "imagenet-c":
        if args.corruption is None:
            raise ValueError("--corruption is required for ImageNet-C")
        imagenet_root = Path(args.data_root) / "ImageNet-C"
        dataset = TinyImageNetCorruptionDataset(
            root=imagenet_root,
            corruption=args.corruption,
            severity=args.severity,
            transform=sd_transform,
        )
        # Human-readable ImageNet-1K labels, falling back to WNIDs
        clean_root = Path(args.data_root) / "ImageNet-1000"
        wnid_to_label = _load_imagenet_wnid_to_label(clean_root)
        class_names = []
        for i in range(len(dataset.idx_to_class)):
            wnid = dataset.idx_to_class[i]
            class_names.append(wnid_to_label.get(wnid, wnid))
    else:
        raise ValueError(
            "Classifier eval mode is only supported for CIFAR-C, Tiny-ImageNet-C, and ImageNet-C datasets"
        )

    # Optionally restrict evaluation to a small subset for speed
    if getattr(args, "overfit_dset_size", 0) and args.overfit_dset_size > 0:
        max_size = min(args.overfit_dset_size, len(dataset))
        dataset = Subset(dataset, torch.arange(max_size))

    loader = DataLoader(
        dataset,
        batch_size=min(args.loader_batch_size, 64),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # CLIP-based classifier: we treat CLIP logits over text prompts as class logits
    clip_reward = CLIPReward(
        class_names=class_names,
        device=device,
        reward_variant="logit_max_margin",
        model_name=args.clip_variant,
        lpips_weight=0.0,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in tqdm(loader, desc="Evaluating CLIP classifier", leave=False):
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Reuse CLIPReward's internal helper to get logits
        logits, _ = clip_reward._compute_target_logits(images, labels)  # type: ignore[attr-defined]
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)

    print(
        {
            "mode": "eval_classifier",
            "dataset": args.dataset,
            "corruption": args.corruption,
            "severity": args.severity,
            "loss": avg_loss,
            "acc": acc,
        }
    )

