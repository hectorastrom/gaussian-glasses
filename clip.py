import torch
from transformers import pipeline
from PIL import Image

clip = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    dtype=torch.bfloat16,
    use_fast=True,
    device=0
)

labels=["a photo of a man", "a photo of a dog", "a photo of a room", "a photo of a landscape"]
img_name = "output_img2img.png"
source_img = Image.open(img_name)
print(clip(source_img, labels))
