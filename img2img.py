import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from torchvision import transforms

# Load pipeline
device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

# input_image_path = "outputs/output_text2img.png"
# datasets/COD10K/Train/Image/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg
input_image_path = input("Input image: ")
init_image = load_image(input_image_path).convert("RGB")

init_image = init_image.resize((512, 512))
# add white background
# white_bg = Image.new("RGBA", init_image.size, "WHITE")
# white_bg.paste(init_image, (0,0), init_image)
# result_img = white_bg.convert("RGB")

# result_img.save("intermediate.png")

prompt = input("Modification from input image: ")
# strength determines how much of original image is modified (0=None, 1=New)
strength = 0.6

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=7.5 # how strongly to follow text prompt
).images[0]

to_tensor = transforms.ToTensor()
tensor_img = to_tensor(image)
print(tensor_img.min(), tensor_img.max())

image.save("outputs/output_img2img.png")
