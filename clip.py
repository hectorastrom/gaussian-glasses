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

print("Selecting from a predefined set of labels...")
labels=["a photo of a man", "a photo of a dog", "a photo of a room", "a photo of a landscape", "a photo of a coffee cup"]

img_name = "outputs/output_img2img.png"
source_img = Image.open(img_name)
# sorted by score list of dict [{'score': float, 'label': str}, ...] for all labels
likelihoods = clip(source_img, labels) 
print(f"Most likely: '{likelihoods[0]['label']}' with confidence: {likelihoods[0]['score']:.2f}")
