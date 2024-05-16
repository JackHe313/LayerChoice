import os
import torch
import numpy as np
from datasets import load_dataset
from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image

dataset = load_dataset("cifar10")
save_dir = "./processed_cifar10"

name_count = 0

def save_image(img_array, base_save_path, class_label):
    global name_count
    class_folder = os.path.join(base_save_path, str(class_label))
    os.makedirs(class_folder, exist_ok=True)  # Create the class folder if it doesn't exist
    img = Image.fromarray(img_array)
    img.save(os.path.join(class_folder, f"{name_count}.png"))
    name_count+=1

def setup_pipeline():
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    return pipe

def transform_cifar_class(element, pipe):
    current_class = element['label']
    target_class = class_mapping[current_class]
    source_prompt = prompt_mapping[current_class]
    target_prompt = prompt_mapping[target_class]
    print(source_prompt, target_prompt)
    init_image = element['img'].resize((768, 768))

    mask_image = pipe.generate_mask(image=init_image, source_prompt=source_prompt, target_prompt=target_prompt)
    image_latents = pipe.invert(image=init_image, prompt=source_prompt).latents
    transformed_image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]
    
    save_image(np.array(transformed_image), os.path.join(save_dir, 'diffedit'), target_class)
    element['img'] = np.array(transformed_image)
    element['label'] = target_class
    return element

# Parameters
class_mapping = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5,
    5: 6, 6: 7, 7: 8, 8: 9, 9: 0 
}
prompt_mapping = {
    0: "an image of a plane",
    1: "an image of a car",
    2: "an image of a bird",
    3: "an image of a cat",
    4: "an image of a deer",
    5: "an image of a dog",
    6: "an image of a frog",
    7: "an image of a horse",
    8: "an image of a ship",
    9: "an image of a truck",
}

#single sample
##########################################
pipe = setup_pipeline()
transform_cifar_class(dataset['train'][2], pipe)

#whole dataset transform
##########################################
# pipe = setup_pipeline()
# transformed_dataset = dataset['train'].map(lambda x: transform_cifar_class(x, pipe))
# transformed_dataset.save_to_disk("./processed_cifar10/transformed_dataset")

