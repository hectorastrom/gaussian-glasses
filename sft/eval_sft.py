# @Time    : 2025-12-01
# @Author  : Kevin Zhu
# @File    : eval_sft.py
"""
Evaluate SFT-trained LoRA checkpoints on the COD10K test set.

Measures classification accuracy before and after SD enhancement.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPModel, CLIPTokenizerFast
from peft import PeftModel

from data.COD_dataset import build_COD_torch_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT-trained models")
    
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--strength", type=float, default=0.4,
                        help="I2I noise strength (should match training)")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_oracle_prompt", action="store_true", default=True)
    parser.add_argument("--no_oracle_prompt", dest="use_oracle_prompt", 
                        action="store_false")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save before/after sample images")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for faster testing)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = "cuda"
    sd_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    clip_model_id = "openai/clip-vit-base-patch16"
    
    print("=" * 60)
    print("Evaluating Gradient-Trained Model")
    print("=" * 60)
    print(f"LoRA path: {args.lora_path}")
    print(f"Strength: {args.strength}, Guidance: {args.guidance}")
    print()
    
    # ========== Load Test Dataset ==========
    print("Loading test dataset...")
    test_set = build_COD_torch_dataset(split_name="test")
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None and args.max_samples > 0:
        test_set = Subset(test_set, range(min(args.max_samples, len(test_set))))
        # Preserve metadata on subset
        test_set.all_classes = test_set.dataset.all_classes
        test_set.label2str = test_set.dataset.label2str
        print(f"Limited to {len(test_set)} samples for faster evaluation")
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # ========== Load SD Pipeline with LoRA ==========
    print("Loading Stable Diffusion pipeline...")
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float16,
    ).to(device)
    sd_pipe.set_progress_bar_config(disable=True)
    
    # Load gradient-trained LoRA (peft format)
    print(f"Loading LoRA from {args.lora_path}...")
    sd_pipe.unet = PeftModel.from_pretrained(sd_pipe.unet, args.lora_path)
    sd_pipe.unet.eval()
    
    # ========== Setup CLIP ==========
    print("Loading CLIP classifier...")
    clip_model = CLIPModel.from_pretrained(
        clip_model_id, torch_dtype=torch.float16
    ).to(device)
    clip_model.eval()
    
    tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_id)
    
    # Pre-compute class embeddings
    class_names = test_set.all_classes
    prompts = [f"An image of {label}" for label in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)
    
    # CLIP normalization
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    
    def preprocess_for_clip(images):
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        return ((images - clip_mean) / clip_std).to(torch.float16)
    
    # ========== Evaluation Loop ==========
    MAX_N = 5
    correct_orig = [0] * MAX_N
    correct_gen = [0] * MAX_N
    total = 0
    
    print("\nRunning evaluation...")
    pbar = tqdm(test_loader, desc="Evaluating")
    
    last_orig = None
    last_gen = None
    
    for batch in pbar:
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        batch_size = labels.size(0)
        
        # Build prompts
        if args.use_oracle_prompt:
            batch_prompts = [
                f"A clear photo of {test_set.label2str(l.item())}" 
                for l in labels
            ]
        else:
            batch_prompts = [""] * batch_size
        
        # Generate enhanced images
        with torch.no_grad():
            generations = sd_pipe(
                batch_prompts,
                images,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.num_steps,
                output_type="pt",
            ).images
        
        # CLIP evaluation
        with torch.no_grad():
            # Original images
            orig_norm = preprocess_for_clip(images)
            orig_features = clip_model.get_image_features(pixel_values=orig_norm)
            orig_features = F.normalize(orig_features, dim=-1)
            logits_orig = clip_model.logit_scale.exp() * (orig_features @ text_features.t())
            
            # Generated images
            gen_norm = preprocess_for_clip(generations)
            gen_features = clip_model.get_image_features(pixel_values=gen_norm)
            gen_features = F.normalize(gen_features, dim=-1)
            logits_gen = clip_model.logit_scale.exp() * (gen_features @ text_features.t())
        
        # Compute accuracy
        _, top_n_orig = logits_orig.topk(MAX_N, dim=1)
        _, top_n_gen = logits_gen.topk(MAX_N, dim=1)
        
        target_labels = labels.view(-1, 1)
        matches_orig = (top_n_orig == target_labels)
        matches_gen = (top_n_gen == target_labels)
        
        for k in range(MAX_N):
            correct_orig[k] += matches_orig[:, :k+1].any(dim=1).sum().item()
            correct_gen[k] += matches_gen[:, :k+1].any(dim=1).sum().item()
        
        total += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            "orig_acc": f"{100*correct_orig[0]/total:.1f}%",
            "gen_acc": f"{100*correct_gen[0]/total:.1f}%",
        })
        
        # Save samples for visualization
        last_orig = images[0]
        last_gen = generations[0]
    
    # ========== Results ==========
    print("\n" + "=" * 60)
    print(f"{'Metric':<12} | {'Original':<15} | {'Generated':<15} | {'Δ':<10}")
    print("-" * 60)
    
    for n in range(MAX_N):
        acc_orig = correct_orig[n] / total
        acc_gen = correct_gen[n] / total
        delta = acc_gen - acc_orig
        sign = "+" if delta >= 0 else ""
        print(f"Top-{n+1:<7} | {acc_orig:<15.2%} | {acc_gen:<15.2%} | {sign}{delta:.2%}")
    
    print("=" * 60)
    
    # Save sample images
    if args.save_samples and last_orig is not None:
        import os
        os.makedirs("outputs", exist_ok=True)
        to_pil_image(last_orig.cpu()).save("outputs/eval_gradient_before.png")
        to_pil_image(last_gen.cpu()).save("outputs/eval_gradient_after.png")
        print("\nSaved sample images to outputs/")


if __name__ == "__main__":
    main()


