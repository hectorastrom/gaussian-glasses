import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

# Load pipeline
device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

input_image_path = "output_text2img.png"
init_image = load_image(input_image_path).convert("RGB")

init_image = init_image.resize((512, 512))
# add white background
# white_bg = Image.new("RGBA", init_image.size, "WHITE")
# white_bg.paste(init_image, (0,0), init_image)
# result_img = white_bg.convert("RGB")

# result_img.save("intermediate.png")

prompt = "Oil painting of a man pouring tea for himself at an old table in a colonial household"
# strength determines how much of original image is modified (0=None, 1=New)
strength = 0.6

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=7.5 # how strongly to follow text prompt
).images[0]

image.save("output_img2img.png")
