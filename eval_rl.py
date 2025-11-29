# @Time    : 2025-11-29 13:12
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : eval_rl.py
"""
Goal: Evaluate the performance of the diffusion + CLIP combo on the COD dataset.
"""
from COD_dataset import build_COD_torch_dataset
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
prompt = ""

# where to find weights
# download weights with e.g. aws s3 cp s3://hectorastrom-dl-final/checkpoints/robust-totem-89/epoch224/pytorch_lora_weights.safetensors weights/robust-totem-89/epoch224/lora.safetensors
wandb_runname = "robust-totem-89"
epoch_num = 224
epoch_str = f"/epoch{epoch_num}/" if epoch_num is not None else ""
lora_path = f"./weights/{wandb_runname}{epoch_str}lora.safetensors"

before_image_name = f"before_generated_img_{epoch_num}.png"
after_image_name = f"generated_img_{epoch_num}.png"

###############################
# Load LoRA to SD1.5
###############################
sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16,
).to(device)

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
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
progress_bar = tqdm(test_loader, desc="Benchmarking Test Set", total=len(test_loader))

correct_count = 0
total_count = 0

# pull much of this from img2img.py and clip_classifier.py
for item in progress_bar:
    # 1. SD stage
    # (C, H, W) = (3, 512, 512) if batch_size is 1
    img_tensors = item["pixel_values"].to(sd_pipe.device)
    label_idx = item["label"].item()
    
    # Save original
    pil_img = to_pil_image(img_tensors[0])
    pil_img.save(f"outputs/{before_image_name}")
    
    # Generate - using output_type='pt' to keep it as tensor on GPU for CLIP
    generations = sd_pipe(
        prompt,
        img_tensors,
        strength=strength,
        guidance_scale=guidance,
        # This should be 50, as it is in rl_trainer. The pipeline knows the
        # first denoising timestep should be at effective timestep 
        # 50 * (1.0 - strength). 
        num_inference_steps=50, 
        output_type='pt'
    ).images
    
    # 2. CLIP stage
    # Prepare images: Resize to 224x224 and Normalize
    images = generations # (1, 3, 512, 512)
    images_resized = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    images_norm = (images_resized - clip_mean) / clip_std

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=images_norm.to(dtype=torch.float16))
        image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        logit_scale = clip_model.logit_scale.exp()
        # Shape: (Batch, Num_Classes)
        logits = logit_scale * (image_features @ text_features.t())
        
        pred_idx = logits.argmax(dim=-1).item()

    # Metrics
    is_correct = (pred_idx == label_idx)
    correct_count += int(is_correct)
    total_count += 1
    
    tqdm.write(f"True: {class_names[label_idx]} | Pred: {class_names[pred_idx]} | Correct: {is_correct}")
    tqdm.write(f"Running Accuracy: {correct_count/total_count:.2%}")

    # Save output (converting tensor back to PIL)
    gen_image = to_pil_image(generations[0])
    gen_image.save(f'outputs/{after_image_name}')
    
    # Just for testing first iteration
    break