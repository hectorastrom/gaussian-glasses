# @Time    : 2025-12-01
# @Author  : Kevin Zhu
# @File    : sft_trainer.py
"""
Gradient-based training for camouflage task.

Instead of RL (DDPO), we directly backprop through:
    LoRA UNet → Diffusion → VAE Decode → Frozen CLIP → Classification Loss

This gives us exact gradients (lower variance than policy gradients).

Usage: 
python sft_trainer.py --epochs 30 --batch_size 2 --overfit_size 0 --num_steps 10 --guidance_scale 1.0 --prompt ORACLE

"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPTokenizerFast
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb

from data.COD_dataset import build_COD_torch_dataset


# ==========================================
# Differentiable Denoising Loop
# ==========================================

def differentiable_denoise(
    unet: nn.Module,
    scheduler: DDIMScheduler,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    start_step_ratio: float = 0.4,
) -> torch.Tensor:
    """
    Deterministic DDIM denoising loop with full gradient support.
    
    Args:
        unet: The UNet model (with LoRA)
        scheduler: DDIM scheduler (must use eta=0 for determinism)
        latents: Noisy latents to denoise (B, 4, H/8, W/8)
        prompt_embeds: Text embeddings for conditioning (B*2, 77, 768) if CFG
        num_steps: Total diffusion steps
        guidance_scale: CFG scale (>1 enables classifier-free guidance)
        start_step_ratio: What fraction of steps to actually run (for I2I)
    
    Returns:
        Denoised latents with gradients attached
    """
    device = latents.device
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    
    # For I2I: skip early steps, start from partially noised latent
    start_idx = int(len(timesteps) * (1 - start_step_ratio))
    timesteps = timesteps[start_idx:]
    
    do_cfg = guidance_scale > 1.0
    batch_size = latents.shape[0]
    
    for t in timesteps:
        # Duplicate latents for CFG (unconditional + conditional)
        if do_cfg:
            latent_input = torch.cat([latents, latents])
            # prompt_embeds already has [uncond, cond] format
            prompt_embeds_to_use = prompt_embeds
        else:
            latent_input = latents
            # When CFG disabled, only use conditional embeddings (second half)
            prompt_embeds_to_use = prompt_embeds[batch_size:]
        
        latent_input = scheduler.scale_model_input(latent_input, t)
        
        # UNet forward pass - gradients flow through LoRA layers
        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds_to_use,
            return_dict=False,
        )[0]
        
        # Apply classifier-free guidance
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # DDIM step with eta=0 (deterministic, fully differentiable)
        latents = scheduler.step(
            noise_pred, t, latents, 
            eta=0.0, 
            return_dict=False
        )[0]
    
    return latents


# ==========================================
# Main Trainer Class
# ==========================================

class GradientCamouflageTrainer:
    """
    Trains SD LoRA to improve CLIP classification via direct gradient descent.
    
    Pipeline: Image → VAE → Noise → UNet (LoRA) → VAE → CLIP → Loss
    """
    
    def __init__(
        self,
        sd_model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        clip_model_name: str = "openai/clip-vit-base-patch16",
        lora_rank: int = 4,
        lora_alpha: int = 4,
        device: str = "cuda",
    ):
        self.device = device
        print(f"Initializing on device: {device}")
        
        # ---------- Load SD Components ----------
        print("Loading Stable Diffusion components...")
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_name, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_name, subfolder="unet", torch_dtype=torch.float16
        ).to(device)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model_name, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_model_name, subfolder="tokenizer"
        )
        
        self.scheduler = DDIMScheduler.from_pretrained(
            sd_model_name, subfolder="scheduler"
        )
        
        # ---------- Setup LoRA on UNet ----------
        print("Applying LoRA to UNet attention layers...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        # Freeze non-LoRA params explicitly
        for name, param in self.unet.named_parameters():
            param.requires_grad = "lora" in name.lower()
        
        # Freeze VAE and text encoder entirely
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        
        # ---------- Load Frozen CLIP Classifier ----------
        print(f"Loading CLIP classifier: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(
            clip_model_name, torch_dtype=torch.float32  # FP32 for stable grads
        ).to(device)
        self.clip.eval()
        self.clip.requires_grad_(False)
        
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_name)
        
        # CLIP normalization constants
        self.clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1)
        
        # Will cache text embeddings for all classes
        self.class_text_embeds = None
        self.class_names = None
        
        print("Initialization complete!\n")
    
    def cache_class_embeddings(self, class_names: list[str]):
        """Pre-compute normalized CLIP text embeddings for all classes."""
        self.class_names = class_names
        prompts = [f"An image of {name}" for name in class_names]
        
        inputs = self.clip_tokenizer(
            prompts, padding=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip.get_text_features(**inputs)
            self.class_text_embeds = F.normalize(text_features, dim=-1)
        
        print(f"Cached embeddings for {len(class_names)} classes")
    
    def preprocess_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize images for CLIP.
        
        Args:
            images: (B, 3, H, W) in range [0, 1]
        Returns:
            (B, 3, 224, 224) normalized for CLIP
        """
        images = F.interpolate(
            images, size=(224, 224), mode="bicubic", align_corners=False
        )
        images = (images - self.clip_mean) / self.clip_std
        return images
    
    def encode_prompts(self, prompts: list[str], batch_size: int) -> torch.Tensor:
        """
        Encode text prompts for SD conditioning (with CFG support).
        
        Returns concatenated [uncond_embeds, cond_embeds] for CFG.
        """
        # Conditional embeddings
        text_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            cond_embeds = self.text_encoder(text_ids)[0]
        
        # Unconditional embeddings (empty prompts)
        uncond_ids = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            uncond_embeds = self.text_encoder(uncond_ids)[0]
        
        # CFG format: [uncond, cond]
        return torch.cat([uncond_embeds, cond_embeds], dim=0)
    
    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        prompts: list[str],
        noise_strength: float = 0.4,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full differentiable forward pass through the pipeline.
        
        Args:
            images: Input images (B, 3, H, W) in [0, 1]
            labels: Ground truth class indices (B,)
            prompts: Text prompts for SD conditioning
            noise_strength: Fraction of diffusion to run (0.4 = last 40%)
            num_steps: Total diffusion steps
            guidance_scale: Classifier-free guidance scale
        
        Returns:
            loss: Cross-entropy classification loss
            accuracy: Batch classification accuracy  
            generated: Generated images (B, 3, H, W) in [0, 1]
            preds: Predicted class indices (B,)
        """
        batch_size = images.shape[0]
        
        # ---------- 1. VAE Encode ----------
        images_scaled = 2.0 * images - 1.0  # [0,1] → [-1,1]
        with torch.no_grad():
            latents = self.vae.encode(
                images_scaled.to(self.vae.dtype)
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # ---------- 2. Add Noise (for I2I) ----------
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        start_idx = int(len(timesteps) * (1 - noise_strength))
        t_start = timesteps[start_idx]
        
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(
            latents, noise, t_start.expand(batch_size)
        )
        noisy_latents = noisy_latents.to(self.unet.dtype)
        
        # ---------- 3. Encode Prompts ----------
        prompt_embeds = self.encode_prompts(prompts, batch_size)
        
        # ---------- 4. Differentiable Denoising ----------
        self.unet.train()
        denoised_latents = differentiable_denoise(
            unet=self.unet,
            scheduler=self.scheduler,
            latents=noisy_latents,
            prompt_embeds=prompt_embeds,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            start_step_ratio=noise_strength,
        )
        
        # ---------- 5. VAE Decode ----------
        denoised_latents = denoised_latents.to(self.vae.dtype)
        denoised_latents = denoised_latents / self.vae.config.scaling_factor
        
        # VAE decode is differentiable (frozen weights, but grads flow through)
        generated = self.vae.decode(denoised_latents, return_dict=False)[0]
        generated = (generated + 1.0) / 2.0  # [-1,1] → [0,1]
        generated = generated.clamp(0, 1)
        
        # ---------- 6. CLIP Classification ----------
        clip_input = self.preprocess_for_clip(generated.float())
        image_features = self.clip.get_image_features(pixel_values=clip_input)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute logits against cached class embeddings
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * (image_features @ self.class_text_embeds.t())
        
        # ---------- 7. Classification Loss ----------
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy for logging
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        
        return loss, accuracy, generated, preds
    
    def validate(
        self,
        val_loader,
        noise_strength: float = 0.4,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        prompt: str = None,
        dataset=None,
    ) -> dict:
        """
        Run validation on the validation dataset.
        
        Returns:
            Dictionary with validation metrics: loss, accuracy
        """
        self.unet.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)
                
                if prompt == "ORACLE":
                    prompts = [
                        f"A clear photo of {dataset.label2str(l.item())}" 
                        for l in labels
                    ]
                elif prompt is not None:
                    prompts = [prompt] * len(labels)
                else:
                    prompts = [""] * len(labels)
                
                with autocast():
                    loss, accuracy, generated, preds = self.forward(
                        images=images,
                        labels=labels,
                        prompts=prompts,
                        noise_strength=noise_strength,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                    )
                
                total_loss += loss.item()
                total_acc += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        self.unet.train()  # Switch back to training mode
        
        return {
            "val/loss": avg_loss,
            "val/accuracy": avg_acc,
        }
    
    def train(
        self,
        dataset,
        val_dataset=None,
        num_epochs: int = 50,
        batch_size: int = 2,
        lr: float = 1e-4,
        noise_strength: float = 0.4,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        prompt: str = None,
        gradient_accumulation_steps: int = 8,
        save_every: int = 10,
        val_every: int = 100,
        log_wandb: bool = True,
        image_size: int = 512,
        overfit_size: int = -1,
    ):
        """
        Main training loop.
        
        Args:
            dataset: COD dataset with .all_classes and .label2str
            val_dataset: Validation dataset (test split). If None, validation is skipped.
            num_epochs: Number of training epochs
            batch_size: Per-device batch size (keep small for memory)
            lr: Learning rate for LoRA parameters
            noise_strength: I2I noise strength (0.4 = modify last 40%)
            num_steps: Diffusion steps (fewer = faster, less quality)
            guidance_scale: CFG scale
            prompt: Prompt for diffusion model. If "ORACLE", uses ground-truth label. If None, uses empty prompt.
            gradient_accumulation_steps: Effective batch = batch_size * this
            save_every: Save checkpoint every N epochs
            val_every: Run validation every N optimizer steps
            log_wandb: Whether to log to Weights & Biases
        """
        # Cache class embeddings
        self.cache_class_embeddings(dataset.all_classes)
        
        # Collect LoRA parameters
        lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        print(f"Training {sum(p.numel() for p in lora_params):,} LoRA parameters")
        
        optimizer = torch.optim.AdamW(lora_params, lr=lr)
        scaler = GradScaler()
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        # Setup validation loader if validation dataset provided
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            print(f"Validation dataset: {len(val_dataset)} samples")
        
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"./gradient_logs/{run_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        if log_wandb:
            wandb_config = {
                # Training hyperparameters
                "lr": lr,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch": batch_size * gradient_accumulation_steps,
                "num_epochs": num_epochs,
                
                # Diffusion parameters
                "noise_strength": noise_strength,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                
                # Dataset parameters
                "image_size": image_size,
                "train_dataset_size": len(dataset),
                "overfit_size": overfit_size if overfit_size > 0 else None,
                "val_dataset_size": len(val_dataset) if val_dataset is not None else None,
                
                # Prompt settings
                "prompt": prompt if prompt is not None else "empty",
                
                # Training settings
                "save_every": save_every,
                "val_every": val_every,
                
                # Model architecture
                "lora_rank": 4,
                "lora_alpha": 4,
            }
            wandb.init(
                project="camouflage-gradient",
                name=run_name,
                config=wandb_config
            )
        
        global_step = 0
        running_loss = 0.0
        running_loss_count = 0
        running_acc = 0.0  # Accumulate accuracy
        running_acc_count = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                images = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)
                
                if prompt == "ORACLE":
                    prompts = [
                        f"A clear photo of {dataset.label2str(l.item())}" 
                        for l in labels
                    ]
                elif prompt is not None:
                    prompts = [prompt] * len(labels)
                else:
                    prompts = [""] * len(labels)
                
                with autocast():
                    loss, accuracy, generated, preds = self.forward(
                        images=images,
                        labels=labels,
                        prompts=prompts,
                        noise_strength=noise_strength,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                    )
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if log_wandb and global_step % 10 == 0:
                        # Log running average over accumulated batches
                        avg_running_loss = running_loss / running_loss_count if running_loss_count > 0 else 0.0
                        avg_running_acc = running_acc / running_acc_count if running_acc_count > 0 else 0.0
                        log_dict = {
                            "train/loss": avg_running_loss,
                            "train/accuracy": avg_running_acc,  # Average over accumulated batches
                            "train/step": global_step,
                        }
                        
                        # Reset running average after logging
                        running_loss = 0.0
                        running_loss_count = 0
                        running_acc = 0.0
                        running_acc_count = 0
                        
                        # Log images periodically (every 50 steps)
                        if global_step % 50 == 0:
                            # Convert images to numpy for wandb (B, 3, H, W) -> (H, W, 3) for display
                            # Take first image from batch for visualization
                            with torch.no_grad():
                                orig_img = images[0].cpu().permute(1, 2, 0).numpy()
                                gen_img = generated[0].cpu().permute(1, 2, 0).numpy()
                                
                                # Ensure values are in [0, 1] range
                                orig_img = orig_img.clip(0, 1)
                                gen_img = gen_img.clip(0, 1)
                                
                                true_label = dataset.label2str(labels[0].item())
                                pred_label = dataset.label2str(preds[0].item())
                                log_dict["images/original"] = wandb.Image(orig_img, caption=f"Original (label: {true_label})")
                                log_dict["images/generated"] = wandb.Image(gen_img, caption=f"Generated (pred: {pred_label}, true: {true_label})")
                        
                        wandb.log(log_dict, step=global_step)
                    
                    # Run validation periodically (only when global_step increments)
                    if val_loader is not None and global_step > 0 and global_step % val_every == 0:
                        val_metrics = self.validate(
                            val_loader=val_loader,
                            noise_strength=noise_strength,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                            prompt=prompt,
                            dataset=val_dataset,
                        )
                        print(f"\n[Step {global_step}] Validation - Loss: {val_metrics['val/loss']:.4f}, Acc: {100*val_metrics['val/accuracy']:.1f}%\n")
                        if log_wandb:
                            val_metrics["train/step"] = global_step
                            wandb.log(val_metrics, step=global_step)
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_acc += accuracy.item()
                num_batches += 1
                
                # Update running averages
                running_loss += loss.item() * gradient_accumulation_steps
                running_loss_count += 1
                running_acc += accuracy.item()  # Accumulate accuracy
                running_acc_count += 1
                
                pbar.set_postfix({
                    "loss": f"{epoch_loss/num_batches:.4f}",
                    "acc": f"{100*epoch_acc/num_batches:.1f}%",
                })
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Acc={100*avg_acc:.1f}%\n")
            
            if log_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "epoch/loss": avg_loss,
                    "epoch/accuracy": avg_acc,
                }
                
                # Run validation at end of each epoch
                if val_loader is not None:
                    val_metrics = self.validate(
                        val_loader=val_loader,
                        noise_strength=noise_strength,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        prompt=prompt,
                        dataset=val_dataset,
                    )
                    print(f"Epoch {epoch+1} Validation - Loss: {val_metrics['val/loss']:.4f}, Acc: {100*val_metrics['val/accuracy']:.1f}%\n")
                    log_dict.update(val_metrics)
                    log_dict["epoch/val_loss"] = val_metrics['val/loss']
                    log_dict["epoch/val_accuracy"] = val_metrics['val/accuracy']
                
                wandb.log(log_dict, step=global_step)
            
            if (epoch + 1) % save_every == 0:
                ckpt_dir = os.path.join(save_dir, f"epoch{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.unet.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint: {ckpt_dir}")
        
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.unet.save_pretrained(final_dir)
        print(f"Training complete! Final model: {final_dir}")
        
        if log_wandb:
            wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradient-based camouflage training (no RL)"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Keep small (1-2) due to memory")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Effective batch = batch_size * this")
    
    # Diffusion
    parser.add_argument("--noise_strength", type=float, default=0.4,
                        help="I2I noise (0.4 = modify last 40%% of steps)")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Diffusion steps (fewer = faster, more memory efficient)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    # Dataset
    parser.add_argument("--overfit_size", type=int, default=128,
                        help="Subset size for testing. Set <=0 for full dataset")
    parser.add_argument("--image_size", type=int, default=512)
    
    # Prompts
    parser.add_argument("--prompt", type=str, default=None,
                        help='Prompt for diffusion model. Use "ORACLE" for ground-truth label, or provide custom prompt. If not set, uses empty prompt.')
    
    # Logging
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=100,
                        help="Run validation every N optimizer steps")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Gradient-Based Camouflage Training")
    print("=" * 60)
    print(f"Config: {vars(args)}\n")
    
    print("Loading COD10K dataset...")
    train_dataset = build_COD_torch_dataset("train", image_size=args.image_size)
    
    if args.overfit_size > 0:
        train_dataset = Subset(train_dataset, range(min(args.overfit_size, len(train_dataset))))
        train_dataset.all_classes = train_dataset.dataset.all_classes
        train_dataset.label2str = train_dataset.dataset.label2str
        print(f"Using subset of {len(train_dataset)} samples for overfitting test")
    else:
        print(f"Using full dataset: {len(train_dataset)} samples")
    
    # Load validation dataset (test split)
    val_dataset = None
    if args.val_every > 0:
        print("Loading validation dataset (test split)...")
        val_dataset = build_COD_torch_dataset("test", image_size=args.image_size)
        print(f"Validation dataset: {len(val_dataset)} samples")
    
    trainer = GradientCamouflageTrainer()
    
    trainer.train(
        dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_strength=args.noise_strength,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        prompt=args.prompt,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_every=args.save_every,
        val_every=args.val_every,
        log_wandb=not args.no_wandb,
        image_size=args.image_size,
        overfit_size=args.overfit_size,
    )


if __name__ == "__main__":
    main()

