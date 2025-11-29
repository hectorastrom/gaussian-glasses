# @Time    : 2025-11-29 13:12
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : eval_rl.py
"""
Goal: Evaluate the performance of the diffusion + CLIP combo on the COD dataset.
"""
from COD_dataset import build_COD_torch_dataset
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

test_set = build_COD_torch_dataset(split_name="test")

device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

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

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

# loads onto UNet (most likely) - whatever sd_pipeline.get_trainable_layers()
# returns & trained LoRA on top of in DDPO
pipe.load_lora_weights(lora_path)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# pull much of this from img2img.py and clip_classifier.py
for item in test_set:
    # (C, H, W) = (3, 512, 512) if batch_size is 1
    img_tensors = item["pixel_values"].to(pipe.device)
    label = item["label"]  # int
    pil_img = to_pil_image(img_tensors)
    pil_img.save(f"outputs/{before_image_name}")
    print(f"Saved outputs/{before_image_name}")
    generations = pipe(
        prompt,
        img_tensors,
        strength=strength,
        guidance_scale=guidance,
        # This should be 50, as it is in rl_trainer. The pipeline knows the
        # first denoising timestep should be at effective timestep 
        # 50 * (1.0 - strength). 
        num_inference_steps=50, 
        output_type='pil'
    ).images
    break

gen_image = generations[0]
gen_image.save(f'outputs/{after_image_name}')
print(f"Saved outputs/{after_image_name}")

