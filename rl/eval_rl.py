# @Time    : 2025-11-29 13:12
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : eval_rl.py
"""
Goal: Evaluate the performance of the diffusion + CLIP combo on the COD dataset.
"""
from data.COD_dataset import build_COD_torch_dataset
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPModel, CLIPTokenizerFast
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

test_set = build_COD_torch_dataset(split_name="test")

device = "cuda"
sd_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
clip_model_id = "openai/clip-vit-base-patch16"

###############################
# Training run hyperparams
###############################
strength = 0.4
guidance = 7.0
prompt = "" # TODO: currently don't support ORACLE prompt

# where to find weights
# download weights with e.g. aws s3 cp s3://hectorastrom-dl-final/checkpoints/robust-totem-89/epoch224/pytorch_lora_weights.safetensors weights/robust-totem-89/epoch224/lora.safetensors
wandb_runname = "robust-totem-89"
epoch_num = 224
epoch_str = f"/epoch{epoch_num}/" if epoch_num is not None else ""
lora_path = f"./weights/{wandb_runname}{epoch_str}lora.safetensors"

# Only used for saving the last image for inspection
before_image_name = f"before_generated_img_{epoch_num}.png"
after_image_name = f"generated_img_{epoch_num}.png"

###############################
# Load LoRA to SD1.5
###############################
sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16,
).to(device)
sd_pipe.set_progress_bar_config(disable=True) # had to dig deep to find this

# loads onto UNet (most likely) - whatever sd_pipeline.get_trainable_layers()
# returns & trained LoRA on top of in DDPO
sd_pipe.load_lora_weights(lora_path)

###############################
# Setting up CLIP
###############################
clip_model = CLIPModel.from_pretrained(
    clip_model_id,
    torch_dtype=torch.float16,
).to(device)
clip_model.eval()

tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_id)

# Precompute label embeddings (optimization from reward.py)
class_names = test_set.all_classes
prompts = [f"An image of {label}" for label in class_names]
text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)
    # Normalize
    text_features = text_features / (text_features.norm(p=2, dim=-1, keepdim=True) + 1e-6)

# Normalization constants (OpenAI CLIP specific)
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

# Loaders
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
progress_bar = tqdm(test_loader, desc="Benchmarking Test Set", total=len(test_loader))

# Stats tracking
MAX_N = 5
correct_orig = [0] * MAX_N
correct_gen = [0] * MAX_N
total_count = 0

def process_image_for_clip(images_tensor: torch.Tensor):
    """
    Resizes and normalizes a batch of images for CLIP.
    Expects input shape (B, 3, H, W) in [0, 1].
    """
    # Resize to 224x224
    images_resized = F.interpolate(images_tensor, size=(224, 224), mode="bicubic", align_corners=False)
    # Normalize
    images_norm = (images_resized - clip_mean) / clip_std
    return images_norm.to(dtype=torch.float16)

# pull much of this from img2img.py and clip_classifier.py
print("Starting eval... (Est: ~40 mins on 1xA10G")
for item in progress_bar:
    # 1. SD stage
    # (B, C, H, W) = (B, 3, 512, 512)
    # Assumed to be [0, 1] from dataset
    img_tensors = item["pixel_values"].to(sd_pipe.device)
    labels = item["label"].to(device) # Shape: (Batch_Size)
    current_batch_size = labels.size(0)
    
    # Generate - using output_type='pt' to keep it as tensor on GPU for CLIP
    batch_prompts = [prompt] * current_batch_size
    generations = sd_pipe(
        batch_prompts,
        img_tensors,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=50, 
        output_type='pt',
    ).images
    
    # 2. CLIP stage
    with torch.no_grad():
        # --- Evaluate Original Image ---
        orig_norm = process_image_for_clip(img_tensors)
        image_features_orig = clip_model.get_image_features(pixel_values=orig_norm)
        image_features_orig = image_features_orig / (image_features_orig.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        
        logits_orig = (clip_model.logit_scale.exp() * (image_features_orig @ text_features.t()))
        
        # --- Evaluate Generated Image ---
        gen_norm = process_image_for_clip(generations)
        image_features_gen = clip_model.get_image_features(pixel_values=gen_norm)
        image_features_gen = image_features_gen / (image_features_gen.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        
        logits_gen = (clip_model.logit_scale.exp() * (image_features_gen @ text_features.t()))

    # 3. Compute Accuracy Stats (Vectorized)
    # Get top MAX_N indices. Shape: (Batch, MAX_N)
    _, top_n_orig = logits_orig.topk(MAX_N, dim=1) 
    _, top_n_gen = logits_gen.topk(MAX_N, dim=1)   
    
    # Labels shape: (Batch, 1) for broadcasting
    target_labels = labels.view(-1, 1)

    # Boolean matrices where True indicates the prediction matches the label
    # Shape: (Batch, MAX_N)
    matches_orig = (top_n_orig == target_labels)
    matches_gen = (top_n_gen == target_labels)

    # Update counts
    for k in range(MAX_N):
        # A sample is correct at Top-K if ANY of the first K columns (0 to k) are True
        # .any(dim=1) collapses the columns -> (Batch,) boolean
        # .sum() counts the True values
        count_k_orig = matches_orig[:, :k+1].any(dim=1).sum().item()
        count_k_gen = matches_gen[:, :k+1].any(dim=1).sum().item()
        
        correct_orig[k] += count_k_orig
        correct_gen[k] += count_k_gen
            
    total_count += current_batch_size
    
    # Simple running log
    # tqdm.write(f"Processed {total_count} | Orig Top-1: {correct_orig[0]/total_count:.1%} | Gen Top-1: {correct_gen[0]/total_count:.1%}")

# Save last images for sanity check (taking first of the last batch)
to_pil_image(img_tensors[0]).save(f"outputs/{before_image_name}")
to_pil_image(generations[0]).save(f'outputs/{after_image_name}')

###############################
# Results Table
###############################
print("\n" + "="*50)
print(f"{'Metric':<15} | {'Original Accuracy':<18} | {'SD+CLIP Accuracy':<18}")
print("-" * 55)

for n in range(MAX_N):
    acc_orig = correct_orig[n] / total_count
    acc_gen = correct_gen[n] / total_count
    print(f"{f'Top-{n+1}':<15} | {acc_orig:<18.2%} | {acc_gen:<18.2%}")

print("="*50 + "\n")