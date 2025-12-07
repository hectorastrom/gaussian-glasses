"""Quick evaluation script for CIFAR-10/100-C and Tiny-ImageNet-C.

Usage (uv):
    uv run eval_corruption_classifier.py --dataset cifar10-c --corruption frost --severity 3
    uv run eval_corruption_classifier.py --dataset tiny-imagenet-c --corruption fog --severity 2

If you have a trained checkpoint, pass --checkpoint path/to/weights.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from corruption_datasets import (
    available_cifar_corruptions,
    build_resnet50_classifier,
    make_cifar_c_loader,
    make_tiny_imagenet_c_loader,
    evaluate_classifier,
)


def _list_tiny_corruptions(root: Path) -> List[str]:
    if not root.exists():
        raise FileNotFoundError(f"Tiny-ImageNet-C root not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet-50 on corruption datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10-c", "cifar100-c", "tiny-imagenet-c"],
        required=True,
    )
    parser.add_argument(
        "--corruption",
        required=True,
        help="Corruption name (e.g., gaussian_noise, frost, fog)",
    )
    parser.add_argument("--severity", type=int, default=1, help="Severity level 1-5")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_root", type=Path, default=Path("./datasets"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint (.pth) with state_dict",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Do not use ImageNet-pretrained weights",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze all layers except final fc",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--list_corruptions",
        action="store_true",
        help="List available corruptions and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve dataset-specific settings
    if args.dataset == "cifar10-c":
        num_classes = 10
        dataset_name = "CIFAR-10-C"
        corruption_list = available_cifar_corruptions(args.data_root, dataset_name)
        if args.list_corruptions:
            print("Available corruptions:", ", ".join(corruption_list))
            return
        loader = make_cifar_c_loader(
            root=args.data_root,
            corruption=args.corruption,
            severity=args.severity,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    elif args.dataset == "cifar100-c":
        num_classes = 100
        dataset_name = "CIFAR-100-C"
        corruption_list = available_cifar_corruptions(args.data_root, dataset_name)
        if args.list_corruptions:
            print("Available corruptions:", ", ".join(corruption_list))
            return
        loader = make_cifar_c_loader(
            root=args.data_root,
            corruption=args.corruption,
            severity=args.severity,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    else:
        num_classes = 200
        tiny_root = args.data_root / "Tiny-ImageNet-C"
        corruption_list = _list_tiny_corruptions(tiny_root)
        if args.list_corruptions:
            print("Available corruptions:", ", ".join(corruption_list))
            return
        loader = make_tiny_imagenet_c_loader(
            root=tiny_root,
            corruption=args.corruption,
            severity=args.severity,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    if args.corruption not in corruption_list:
        raise ValueError(
            f"Corruption '{args.corruption}' not found. Available: {', '.join(corruption_list)}"
        )

    pretrained = not args.no_pretrained
    model, _ = build_resnet50_classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=args.device)
        if isinstance(state, Dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys: {unexpected}")
        print(f"Loaded checkpoint from {ckpt_path}")

    metrics = evaluate_classifier(model, loader, device=args.device)
    print(
        {
            "dataset": args.dataset,
            "corruption": args.corruption,
            "severity": args.severity,
            **metrics,
        }
    )


if __name__ == "__main__":
    main()
