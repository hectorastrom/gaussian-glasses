# @Time    : 2025-11-24 15:51
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : rl_trainer.py
# Run with `accelerate launch rl_trainer.py``

from trl import DDPOConfig
from ddpo import ImageDDPOTrainer, I2IDDPOStableDiffusionPipeline
from reward import CLIPReward
from COD_dataset import build_COD_torch_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig
import os

##################################
# Constants
##################################
LOADER_BATCH_SIZE = 16 # limited by CPU - faster to fetch many at once
GPU_BATCH_SIZE = 8 # limited by VRAM
# TODO: Play around with what prompt works best! Can't be class dependent
PROMPT = "A high quality, clear photo"
NUM_WORKERS = 1
NOISE_STRENGTH = 0.6 # controls adherance to original image (1.0 = pure noise, 0.0 = no change)
DEVICE = 'cuda'

##################################
# Construct dataset
##################################
dataset = build_COD_torch_dataset('train')
train_loader = DataLoader(dataset, batch_size=LOADER_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# (called as ... = [self.prompt_fn() for _ in range(batch_size)])
def create_data_generator(dataloader, prompt):
    while True:
        for batch in dataloader:
            images, labels = batch['pixel_values'], batch['label']
            # iterature through batch and yield items one by one
            for i in range(len(images)):
                image = images[i].to(DEVICE)
                label = labels[i].item()
                
                # 3 necessary return items
                metadata = {
                    "label": label,
                    "label_str": dataset.label2str(label),
                    "original_image": image.clone()
                }
                
                # prompt + image are analogous to just prompt within DDPOTrainer
                yield prompt, image, metadata
                

my_generator = create_data_generator(train_loader, PROMPT) 
        
def my_image_loader():
    """
    Return (prompt_str, image_tensor_chw, metadata_dict)
    """
    return next(my_generator)

##################################
# Define reward
##################################
reward_fn = CLIPReward(class_names=dataset.all_classes, device=0)


##################################
# Configuration
##################################
config = DDPOConfig(
    # --- Logging & General ---
    num_epochs=100,
    log_with= "wandb",               # Highly recommended to visualize the Reward curve
    mixed_precision="fp16",         # Standard for SD 1.5
    allow_tf32=True,
    
    # --- Sampling (Experience Collection) ---
    # Total samples per epoch = batch_size * num_batches * num_processes.
    # RL requires a decent "buffer" of experiences to learn effectively.
    # A batch size of 4 fits comfortably on 24GB VRAM with SD 1.5.
    sample_num_steps=10, # 50,
    sample_batch_size=GPU_BATCH_SIZE,           
    sample_num_batches_per_epoch=2, # 8, # 4 * 8 = 32 samples per epoch (per GPU). Good balance.
    
    # --- Training (Update Phase) ---
    train_batch_size=GPU_BATCH_SIZE,       # Must be <= sample_batch_size
    train_gradient_accumulation_steps=1,
    
    # --- Optimizer & LoRA Specifics ---
    # LoRA usually requires a slightly lower LR than full finetuning. 
    # 3e-4 (default) can be unstable; 1e-4 is safer for LoRA.
    train_learning_rate=1e-4,       
    train_use_8bit_adam=True,       # Saves VRAM, virtually no downside.
    
    # --- Critical for Image-to-Image ---
    # In I2I, some images are naturally "harder" to classify than others.
    # You don't want to punish the model for a hard image if it improved 
    # relative to the *last time* it saw that image.
    # This setting tracks a baseline *per prompt*, so we disable it b/c we care
    # about a baseline per image. There's a serious possibility this devastates
    # performance. Leave false, but MAY REVISIT LATER
    per_prompt_stat_tracking=False,  
    
    # --- Project Info ---
    project_kwargs={
        "project_dir": "./ddpo_logs",
        "logging_dir": "./ddpo_logs/runs",
    },
    tracker_project_name="67960-ddpo-classifier-optimization",
)

lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=8,
    lora_alpha=16,
    bias="none",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"] # for UNet TODO: verify
)

pipeline = I2IDDPOStableDiffusionPipeline(
    pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
    use_lora=True,
)

trainer = ImageDDPOTrainer(
    config=config,
    reward_function=reward_fn,
    prompt_function=my_image_loader, # custom prompt function to send images
    sd_pipeline=pipeline,
    noise_strength=NOISE_STRENGTH
)

##################################
# Fix accelerate save_state bug
##################################
# 1. Capture the original method
original_save_state = trainer.accelerator.save_state

# 2. Define a wrapper that forces a default output_dir
def patched_save_state(output_dir=None, **kwargs):
    if output_dir is None:
        # Force the path if it's missing
        output_dir = os.path.join(config.project_kwargs["project_dir"], "checkpoints")
    
    # Ensure the directory exists (the root cause of your error)
    os.makedirs(output_dir, exist_ok=True)
    
    return original_save_state(output_dir=output_dir, **kwargs)

# 3. Apply the patch to the live object
trainer.accelerator.save_state = patched_save_state
print(f"DEBUG: Monkey-patched accelerator.save_state to force output_dir='{config.project_kwargs['project_dir']}'")


##################################
# Begin training!
##################################
if __name__ == "__main__":
    print("Commencing training!")
    trainer.train()