import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler

unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", 
    unet=unet, 
    torch_dtype=torch.float16, 
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
    prompt=prompt, 
    num_inference_steps=4, 
    generator=generator, 
    guidance_scale=8.0
).images[0]

image.save("exploration/outputs/output_lcm.png")