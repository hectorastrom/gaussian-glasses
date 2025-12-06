# @Time    : 2025-11-28 14:00
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : reward.py

import torch
import torch.nn.functional as F
import lpips
from transformers import CLIPModel, CLIPTokenizerFast
from typing import Any, List, Dict

eps = 1e-6

class CLIPReward:
    def __init__(
        self, 
        class_names: List[str], 
        device: str | int = 0, 
        reward_variant: str = "logit_change", 
        model_name: str = None,
        lpips_weight: float = 0.3
    ):
        if reward_variant not in ["logit_max_margin", "logit_change"]:
            raise ValueError(f"Invalid reward_variant: {reward_variant}. Must be 'logit_max_margin' or 'logit_change'.")

        self.reward_variant = reward_variant
        self.device = device
        self.class_names = class_names
        self.lpips_weight = lpips_weight
        
        assert model_name is not None, "Model name is required!"
        self.model_name = model_name

        # --- CLIP Setup ---
        self.clip_model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)

        # Cache text embeddings
        prompts = [f"An image of {label}" for label in class_names]
        text_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / (text_features.norm(p=2, dim=-1, keepdim=True) + eps)
            self.cached_text_embeds = text_features

        # OpenAI CLIP normalization constants
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        # --- LPIPS Setup ---
        # Only load if we are actually using it to save VRAM/compute
        if self.lpips_weight > 0:
            # Alex is fastest - though VGG is best net variant
            self.lpips_func = lpips.LPIPS(net='alex').to(device)
            self.lpips_func.eval()
            for p in self.lpips_func.parameters():
                p.requires_grad_(False)

    def _preprocess_images_clip(self, images: torch.Tensor) -> torch.Tensor:
        """Prepares images for CLIP (Resizing to 224, Normalization)."""
        images = images.to(device=self.device, dtype=torch.float32)
        images = torch.clamp(images, 0.0, 1.0)

        # CLIP ViT-B/32 expects 224x224
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)

        images = (images - self.clip_mean) / self.clip_std
        return images.to(dtype=self.clip_model.dtype)

    def _compute_target_logits(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        pixel_values = self._preprocess_images_clip(images)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + eps)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * (image_features @ self.cached_text_embeds.t())

            labels = labels.to(logits.device)
            target_logits = logits[torch.arange(batch_size, device=logits.device), labels].to(torch.float32)

        return logits, target_logits
        
    def _logit_max_margin_reward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes logit margin from true logit (real class) to highest logit 
        (most confidently predicted class) other than itself
        """
        batch_size = images.shape[0]
        logits, target_logits = self._compute_target_logits(images, labels)
            
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        
        other_logits = logits[mask].view(batch_size, -1) 
        max_other_logits, _ = other_logits.max(dim=1)
        
        rewards = target_logits - max_other_logits
        return rewards
        
    def _logit_change_reward(self, images: torch.Tensor, metadata: List[Dict[str, Any]], labels: torch.Tensor) -> torch.Tensor:
        """Computes change in logits from before / after image"""
        _, target_logits_optimized = self._compute_target_logits(images, labels)
        
        original_images = torch.stack([m["original_image"] for m in metadata], dim=0).to(self.device)
        _, target_logits_original = self._compute_target_logits(original_images, labels)
        
        rewards = target_logits_optimized - target_logits_original
        return rewards

    def _compute_lpips_penalty(self, images: torch.Tensor, metadata: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Computes LPIPS distance between current images and original images.
        Returns a positive tensor where higher values = more difference.
        """
        original_images = torch.stack([m["original_image"] for m in metadata], dim=0).to(self.device)
        
        # LPIPS expects input in range [-1, 1]
        # Current images are [0, 1]
        img_input = (images * 2.0) - 1.0
        img_orig = (original_images * 2.0) - 1.0
        
        # LPIPS forward pass
        with torch.no_grad():
            # Returns (B, 1, 1, 1) -> flatten to (B,)
            lpips_val = self.lpips_func(img_input, img_orig).flatten()
            
        return lpips_val

    def __call__(self, images: torch.Tensor, prompts: List[str], metadata: List[Dict[str, Any]]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            images: (B, C, H, W) Tensor, values [0, 1]
            prompts: List of strings (Ignored)
            metadata: List of dicts containing 'label' and 'original_image'
        """
        # 1. Calculate Base Classification Reward
        labels = torch.tensor([m["label"] for m in metadata], device=self.device)
        
        if self.reward_variant == "logit_max_margin":
            rewards = self._logit_max_margin_reward(images, labels)
        elif self.reward_variant == "logit_change":
            rewards = self._logit_change_reward(images, metadata, labels)
        else:
            raise RuntimeError(f"Unknown reward choice: {self.reward_variant}")
        
        # 2. Apply LPIPS Penalty (if weight is nonzero)
        if self.lpips_weight > 0:
            lpips_loss = self._compute_lpips_penalty(images, metadata)
            
            # Normalize: LPIPS is usually 0.0-0.8. CLIP rewards are approx -5.0
            # to 5.0.
            # Scaling to make magnitudes comparable before weighting
            scaled_lpips = lpips_loss * 6 # e.g. 0.8 * 6 = 4.8
            
            rewards = rewards - (self.lpips_weight * scaled_lpips)

        # 3. Final Cleanup
        rewards = torch.clamp(rewards, -10.0, 10.0)
        if torch.isnan(rewards).any():
            rewards = torch.nan_to_num(rewards, nan=0.0)

        rewards = rewards.to(torch.float32)

        return rewards, {}