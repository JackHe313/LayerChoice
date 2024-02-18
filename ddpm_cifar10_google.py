from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from tqdm import tqdm
import os

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = PNDMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to("cuda")
# loop over number of images you want to generate
number = input('How many generated image do you need?\n')

output_dir = f"ddpm_images/pndm3/pndm_cifar10_32_{number}"
os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(int(number)), desc="Generating Images"):
    # run pipeline in inference (sample random noise and denoise)
    image = ddpm().images[0]

    # save image
    image.save(os.path.join(output_dir, f"pndm_generated_image_{i}.png"))