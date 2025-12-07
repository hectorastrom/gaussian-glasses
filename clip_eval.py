"""
Usage:
python clip_eval.py --eval_type original  # Evaluate on original images
python clip_eval.py --eval_type sd_baseline  # Evaluate on SD-processed images
"""

import torch
import numpy as np
from collections import defaultdict
from transformers import CLIPModel, CLIPTokenizerFast
import torch.nn.functional as F
from PIL import Image
from data.COD_dataset import load_cod10k_lazy
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
import math
import json
import os
import argparse
from tqdm import tqdm

def eval(dataset_split, label_names, clip_model, clip_tokenizer, num_samples=None):
    """
    Evaluate CLIP zero-shot classification on dataset.
    Uses the same CLIP setup as sft_trainer.py for fair comparison.
    """
    if num_samples is None:
        num_samples = len(dataset_split)
    else:
        num_samples = min(num_samples, len(dataset_split))
    
    # CLIP normalization constants (same as sft_trainer.py)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    
    # Pre-compute class text embeddings (same prompt format as sft_trainer.py)
    prompts = [f"An image of {name}" for name in label_names]
    inputs = clip_tokenizer(prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}  # Move to device
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        class_text_embeds = F.normalize(text_features, dim=-1)
    
    device = next(clip_model.parameters()).device
    clip_mean = clip_mean.to(device)
    clip_std = clip_std.to(device)
    class_text_embeds = class_text_embeds.to(device)
    logit_scale = clip_model.logit_scale.exp()
    
    nlls = []
    scores = []
    MAX_K = 5
    correct = [0] * MAX_K  # Track top-1 through top-5 accuracy
    class_nlls = defaultdict(list)
    class_scores = defaultdict(list)
    epsilon = 1e-10

    print(f"Computing NLL and top-{MAX_K} accuracy for {num_samples} samples")

    for i in tqdm(range(num_samples)):
        if i % 100 == 0:
            print(f"Processing sample {i} of {num_samples}")
    
        sample = dataset_split[i]
        image = sample['image']  # PIL Image
        
        # Preprocess image (same as sft_trainer.py)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL to tensor first
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
        # Use bicubic interpolation like sft_trainer.py
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode="bicubic", align_corners=False)
        
        # Normalize (same as sft_trainer.py)
        image_tensor = (image_tensor - clip_mean) / clip_std
        
        # Get image features (same as sft_trainer.py)
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values=image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute logits (same as sft_trainer.py)
            logits = logit_scale * (image_features @ class_text_embeds.t())
            probabilities = F.softmax(logits, dim=-1)
        
        true_label_idx = sample['label']
        true_label_name = label_names[true_label_idx]
        
        # Get score for true label
        score = probabilities[0, true_label_idx].item()
        
        # Get top-k predictions
        _, top_k_preds = logits.topk(MAX_K, dim=1)
        top_k_preds = top_k_preds[0].cpu().numpy()  # Shape: (MAX_K,)
        
        # Check if true label is in top-k for each k
        for k in range(MAX_K):
            if true_label_idx in top_k_preds[:k+1]:
                correct[k] += 1
        
        scores.append(score)
        class_scores[true_label_name].append(score)

        nll = -math.log(score + epsilon)
        nlls.append(nll)
        class_nlls[true_label_name].append(nll)
    
    mean_nll = float(np.mean(nlls))
    std_nll = float(np.std(nlls))
    mean_probability = float(np.mean(scores))
    std_probability = float(np.std(scores))
    
    # Compute top-k accuracies
    accuracies = [float(correct[k] / num_samples) for k in range(MAX_K)]
    
    class_mean_nlls = {k: float(np.mean(v)) for k, v in class_nlls.items()}
    class_mean_scores = {k: float(np.mean(v)) for k, v in class_scores.items()}

    results = {
        'mean_nll': mean_nll,
        'std_nll': std_nll,
        'mean_probability': mean_probability,
        'std_probability': std_probability,
        'class_mean_nlls': class_mean_nlls,
        'class_mean_scores': class_mean_scores,
        'accuracy': accuracies[0],  # Top-1 (for backward compatibility)
        'top1_accuracy': accuracies[0],
        'top2_accuracy': accuracies[1],
        'top3_accuracy': accuracies[2],
        'top4_accuracy': accuracies[3],
        'top5_accuracy': accuracies[4],
        'correct': [int(correct[k]) for k in range(MAX_K)],
        'num_samples': int(num_samples)
    }

    return results

def eval_with_sd_baseline(
    dataset_split, 
    label_names, 
    clip_model, 
    clip_tokenizer,
    sd_pipeline,
    prompt: str = None,
    noise_strength: float = 0.4,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    num_samples=None
):
    """
    Evaluate CLIP on images processed through BASE (non-fine-tuned) Stable Diffusion.
    This provides a baseline to compare against trained models.
    
    Args:
        dataset_split: HuggingFace dataset split
        label_names: List of class names
        clip_model: CLIP model for evaluation
        clip_tokenizer: CLIP tokenizer
        sd_pipeline: Base Stable Diffusion Img2Img pipeline (no LoRA)
        prompt: Prompt for SD. If "ORACLE", uses ground-truth label. If None, uses empty prompt.
        noise_strength: SD noise strength (should match training)
        num_steps: SD inference steps (should match training)
        guidance_scale: SD guidance scale (should match training)
        num_samples: Number of samples to evaluate (None for all)
    
    Returns:
        Dictionary with evaluation metrics
    """
    if num_samples is None:
        num_samples = len(dataset_split)
    else:
        num_samples = min(num_samples, len(dataset_split))
    
    # CLIP normalization constants (same as sft_trainer.py)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    
    # Pre-compute class text embeddings (same prompt format as sft_trainer.py)
    prompts = [f"An image of {name}" for name in label_names]
    inputs = clip_tokenizer(prompts, padding=True, return_tensors="pt")
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        class_text_embeds = F.normalize(text_features, dim=-1)
    
    clip_mean = clip_mean.to(device)
    clip_std = clip_std.to(device)
    class_text_embeds = class_text_embeds.to(device)
    logit_scale = clip_model.logit_scale.exp()
    
    nlls = []
    scores = []
    MAX_K = 5
    correct = [0] * MAX_K
    class_nlls = defaultdict(list)
    class_scores = defaultdict(list)
    epsilon = 1e-10
    
    # Convert PIL images to tensors for SD pipeline
    to_tensor = transforms.ToTensor()
    
    print(f"Evaluating {num_samples} samples with BASE Stable Diffusion...")
    print(f"SD settings: strength={noise_strength}, steps={num_steps}, guidance={guidance_scale}, prompt={prompt if prompt else 'empty'}")
    
    for i in tqdm(range(num_samples), desc="SD+CLIP eval"):
        if i % 100 == 0:
            print(f"Processing sample {i} of {num_samples}")
        
        sample = dataset_split[i]
        image = sample['image']  # PIL Image
        true_label_idx = sample['label']
        true_label_name = label_names[true_label_idx]
        
        # Prepare prompt for SD
        if prompt == "ORACLE":
            sd_prompt = f"A clear photo of {true_label_name}"
        elif prompt is not None:
            sd_prompt = prompt
        else:
            sd_prompt = ""
        
        # Process through BASE Stable Diffusion
        with torch.no_grad():
            # Resize image to 512x512 for SD (standard size)
            image_sd = image.resize((512, 512))
            
            # Generate with SD pipeline
            generated_image = sd_pipeline(
                prompt=sd_prompt,
                image=image_sd,
                strength=noise_strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Convert generated PIL image to tensor for CLIP
            gen_tensor = to_tensor(generated_image).unsqueeze(0).to(device)  # (1, 3, H, W)
            
            # Preprocess for CLIP (same as sft_trainer.py)
            gen_tensor = F.interpolate(gen_tensor, size=(224, 224), mode="bicubic", align_corners=False)
            gen_tensor = (gen_tensor - clip_mean) / clip_std
            
            # Get image features
            image_features = clip_model.get_image_features(pixel_values=gen_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute logits
            logits = logit_scale * (image_features @ class_text_embeds.t())
            probabilities = F.softmax(logits, dim=-1)
        
        # Get score for true label
        score = probabilities[0, true_label_idx].item()
        
        # Get top-k predictions
        _, top_k_preds = logits.topk(MAX_K, dim=1)
        top_k_preds = top_k_preds[0].cpu().numpy()
        
        # Check if true label is in top-k for each k
        for k in range(MAX_K):
            if true_label_idx in top_k_preds[:k+1]:
                correct[k] += 1
        
        scores.append(score)
        class_scores[true_label_name].append(score)
        
        nll = -math.log(score + epsilon)
        nlls.append(nll)
        class_nlls[true_label_name].append(nll)
    
    mean_nll = float(np.mean(nlls))
    std_nll = float(np.std(nlls))
    mean_probability = float(np.mean(scores))
    std_probability = float(np.std(scores))
    
    # Compute top-k accuracies
    accuracies = [float(correct[k] / num_samples) for k in range(MAX_K)]
    
    class_mean_nlls = {k: float(np.mean(v)) for k, v in class_nlls.items()}
    class_mean_scores = {k: float(np.mean(v)) for k, v in class_scores.items()}
    
    results = {
        'mean_nll': mean_nll,
        'std_nll': std_nll,
        'mean_probability': mean_probability,
        'std_probability': std_probability,
        'class_mean_nlls': class_mean_nlls,
        'class_mean_scores': class_mean_scores,
        'accuracy': accuracies[0],
        'top1_accuracy': accuracies[0],
        'top2_accuracy': accuracies[1],
        'top3_accuracy': accuracies[2],
        'top4_accuracy': accuracies[3],
        'top5_accuracy': accuracies[4],
        'correct': [int(correct[k]) for k in range(MAX_K)],
        'num_samples': int(num_samples)
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='CLIP evaluation on COD10K dataset')
    parser.add_argument(
        '--eval_type',
        type=str,
        choices=['original', 'sd_baseline'],
        default='original',
        help='Type of evaluation to run: "original" for original images, "sd_baseline" for SD-processed images'
    )
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = load_cod10k_lazy()

    label_names = dataset['train'].features['label'].names
    print(f"Found {len(label_names)} labels: {label_names[:10]}")

    # Use same CLIP model as sft_trainer.py
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16",  # Same as sft_trainer.py
        torch_dtype=torch.float32
    ).to("cuda")
    clip_model.eval()
    
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")

    # Only load SD pipeline if needed
    sd_pipeline = None
    if args.eval_type == 'sd_baseline':
        print("Loading BASE Stable Diffusion pipeline (no fine-tuning)...")
        sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")
        sd_pipeline.set_progress_bar_config(disable=True)
        sd_pipeline.unet.eval()
        sd_pipeline.vae.eval()

    os.makedirs('eval_outputs', exist_ok=True)

    # Default SD settings (should match training defaults)
    noise_strength = 0.4
    num_steps = 10
    # guidance_scale = 7.5
    guidance_scale = 1.0
    prompt = None  # Use empty prompt by default, can be changed to "ORACLE" or custom prompt

    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Evaluating {split} split")
        print(f"{'='*60}")
        
        if args.eval_type == 'original':
            # Baseline: Original images → CLIP
            print("\nBaseline: Original images → CLIP")
            results_original = eval(dataset[split], label_names, clip_model, clip_tokenizer, num_samples=None)
            
            output_file_orig = f'eval_outputs/eval_results_COD10K_{split}_original.json'
            with open(output_file_orig, 'w') as f:
                json.dump(results_original, f, indent=2)
            
            print(f"Results written to {output_file_orig}")
            print(f"Summary (Original images):")
            print(f"  NLL: {results_original['mean_nll']:.4f}")
            print(f"  Probability: {results_original['mean_probability']:.4f}")
            print(f"  Top-1 Accuracy: {results_original['top1_accuracy']:.4f}")
            print(f"  Top-2 Accuracy: {results_original['top2_accuracy']:.4f}")
            print(f"  Top-3 Accuracy: {results_original['top3_accuracy']:.4f}")
            print(f"  Top-4 Accuracy: {results_original['top4_accuracy']:.4f}")
            print(f"  Top-5 Accuracy: {results_original['top5_accuracy']:.4f}")
        
        elif args.eval_type == 'sd_baseline':
            # Baseline: Original images → BASE SD → CLIP
            print("\nBaseline: Original images → BASE SD → CLIP")
            results_sd_baseline = eval_with_sd_baseline(
                dataset[split], 
                label_names, 
                clip_model, 
                clip_tokenizer,
                sd_pipeline,
                prompt=prompt,
                noise_strength=noise_strength,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                num_samples=None
            )
            
            output_file_sd = f'eval_outputs/eval_results_COD10K_{split}_sd_baseline.json'
            with open(output_file_sd, 'w') as f:
                json.dump(results_sd_baseline, f, indent=2)
            
            print(f"Results written to {output_file_sd}")
            print(f"Summary (BASE SD processed images):")
            print(f"  NLL: {results_sd_baseline['mean_nll']:.4f}")
            print(f"  Probability: {results_sd_baseline['mean_probability']:.4f}")
            print(f"  Top-1 Accuracy: {results_sd_baseline['top1_accuracy']:.4f}")
            print(f"  Top-2 Accuracy: {results_sd_baseline['top2_accuracy']:.4f}")
            print(f"  Top-3 Accuracy: {results_sd_baseline['top3_accuracy']:.4f}")
            print(f"  Top-4 Accuracy: {results_sd_baseline['top4_accuracy']:.4f}")
            print(f"  Top-5 Accuracy: {results_sd_baseline['top5_accuracy']:.4f}")
    
if __name__ == "__main__":
    main()