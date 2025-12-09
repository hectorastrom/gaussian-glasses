# Dataset Downloads

This project supports multiple datasets for training and evaluation:

## COD10K Dataset (Camouflaged Object Detection)
**Download from AWS:**
1. Make sure you're in the project root
1. Download: `aws s3 cp s3://hectorastrom-dl-final/datasets/COD10K-v3.zip ./data/COD.zip`
1. Unzip: `unzip ./data/COD.zip -d ./data/`
1. Rename: `mv data/COD10K-v3 data/COD10K`
1. Clean up: `rm -rf data/COD.zip`

## Corruption Robustness Datasets

For benchmarking model robustness to common corruptions, we support datasets from the paper:
> Benchmarking Neural Network Robustness to Common Corruptions and Perturbations  
> Dan Hendrycks and Thomas Dietterich  
> [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)

### ImageNet-C
- **Download:** [https://zenodo.org/records/2235448](https://zenodo.org/records/2235448)
- **Description:** Full ImageNet-1K (224x224) (1000 classes) with 15 corruption types at 5 severity levels each

### CIFAR-10-C
- **Download:** [https://zenodo.org/records/2535967](https://zenodo.org/records/2535967)
- **Description:** CIFAR-10 (32x32) (10 classes) with 19 corruption types at 5 severity levels each

### Tiny-ImageNet-C
- **Download:** [https://zenodo.org/records/8206060](https://zenodo.org/records/8206060)
- **Description:** Tiny ImageNet (64x64) (200 classes) with corruption types at 5 severity levels each

## Expected Directory Structure

After downloading and extracting, your `data/` directory should look like:

```
data/
├── COD10K/                    # Camouflaged Object Detection dataset
│   ├── Train/
│   ├── Test/
│   └── Info/
├── CIFAR-10-C/               # CIFAR-10 Corruption dataset
│   ├── gaussian_noise.npy
│   ├── shot_noise.npy
│   ├── impulse_noise.npy
│   ├── defocus_blur.npy
│   ├── glass_blur.npy
│   ├── motion_blur.npy
│   ├── zoom_blur.npy
│   ├── snow.npy
│   ├── frost.npy
│   ├── fog.npy
│   ├── brightness.npy
│   ├── contrast.npy
│   ├── elastic_transform.npy
│   ├── pixelate.npy
│   ├── jpeg_compression.npy
│   ├── speckle_noise.npy
│   ├── spatter.npy
│   ├── gaussian_blur.npy
│   ├── saturate.npy
│   └── labels.npy
├── Tiny-ImageNet-C/          # Tiny ImageNet Corruption dataset
│   └── [corruption_name]/
│       └── [severity]/
│           └── [class_id]/
│               └── *.JPEG
├── ImageNet-C/               # Full ImageNet Corruption dataset
│   └── [corruption_name]/
│       └── [severity]/
│           └── [class_id]/
│               └── *.JPEG
```

## Corruption Types Available

All corruption datasets include these common corruption types with 5 severity levels (1-5, where 5 is most severe):

**Noise corruptions:** gaussian_noise, shot_noise, impulse_noise, speckle_noise
**Blur corruptions:** gaussian_blur, defocus_blur, glass_blur, motion_blur, zoom_blur
**Weather corruptions:** snow, frost, fog, brightness
**Digital corruptions:** contrast, elastic_transform, pixelate, jpeg_compression, saturate, spatter

## Usage in Training

Use these datasets with the `--dataset` argument:

```bash
# COD10K (default)
accelerate launch -m rl.rl_trainer

# CIFAR-10-C
accelerate launch -m rl.rl_trainer --dataset cifar10-c --corruption gaussian_noise --severity 1

# CIFAR-100-C
accelerate launch -m rl.rl_trainer --dataset cifar100-c --corruption gaussian_noise --severity 1

# Tiny-ImageNet-C
accelerate launch -m rl.rl_trainer --dataset tiny-imagenet-c --corruption gaussian_noise --severity 1

# ImageNet-C
accelerate launch -m rl.rl_trainer --dataset imagenet-c --corruption gaussian_noise --severity 1
```

For corruption datasets, you must specify `--corruption` and `--severity` arguments.