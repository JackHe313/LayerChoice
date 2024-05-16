import os
import cv2
import torch
import random
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
from torchvision import models
import torchvision.datasets as d
import torchvision.transforms as transforms

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

# Ensure the save directories exist
os.makedirs(save_dir, exist_ok=True)
sub_dirs = ['cropped', 'downsampled', 'edged', 'rotated', 'noisy']
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)


def random_crop_and_zoom(img, crop_size):
    """
    Take a random crop of the image and then resize it back to its original size.
    """
    width, height = img.size
    left = np.random.randint(0, width - crop_size + 1)
    top = np.random.randint(0, height - crop_size + 1)
    right = left + crop_size
    bottom = top + crop_size
    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img.resize((width, height))

def downsample_and_upsample(img, downsample_size):
    """
    Downsample the image to a smaller resolution and then upsample it back to its original size.
    """
    downsampled_img = img.resize((downsample_size, downsample_size))
    return downsampled_img.resize(img.size)

def abstract_representation(img):
    """
    Convert the image to its abstract representation using the Canny edge detector.
    """
    img_np = np.array(img)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    return Image.fromarray(edges)

def random_rotation(img, max_angle):
    """
    Rotate the image by a random angle within [-max_angle, max_angle].
    """
    angle = np.random.uniform(-max_angle, max_angle)
    return img.rotate(angle)

def add_gaussian_noise(img, mean=0, sigma=25):
    """
    Add Gaussian noise to the image.
    """
    img_np = np.array(img)
    row, col, ch = img_np.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_img = np.clip(img_np + gauss, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def split_and_shuffle_grid(img, grid_size):
    """
    Split the image into a grid of the specified size and randomly swap the cells.
    """
    num_rows, num_cols = grid_size
    img_np = np.array(img)
    row_height = img_np.shape[0] // num_rows
    col_width = img_np.shape[1] // num_cols
    
    # List to hold image blocks
    blocks = []
    for r in range(num_rows):
        for c in range(num_cols):
            block = img_np[r*row_height:(r+1)*row_height, c*col_width:(c+1)*col_width]
            blocks.append(block)
    
    # Shuffle the blocks
    random.shuffle(blocks)
    
    # Reconstruct the shuffled image
    shuffled_image = np.zeros_like(img_np)
    idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            shuffled_image[r*row_height:(r+1)*row_height, c*col_width:(c+1)*col_width] = blocks[idx]
            idx += 1
    return Image.fromarray(shuffled_image)

# Modify each function to process an element from the dataset
def random_crop_and_zoom_element(element, crop_size):
    img = element['img']
    processed_img = random_crop_and_zoom(img, crop_size)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'cropped'), class_label)
    element['img'] = np.array(processed_img)
    return element

def downsample_and_upsample_element(element, downsample_size):
    img = element['img']
    processed_img = downsample_and_upsample(img, downsample_size)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'downsampled'), class_label)
    element['img'] = np.array(processed_img)
    return element

def abstract_representation_element(element):
    img = element['img']
    processed_img = abstract_representation(img)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'edged'), class_label)
    element['img'] = np.array(processed_img)
    return element

def random_rotation_element(element, max_angle):
    img = element['img']
    processed_img = random_rotation(img, max_angle)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'rotated'), class_label)
    element['img'] = np.array(processed_img)
    return element

def add_gaussian_noise_element(element, mean=0, sigma=25):
    img = element['img']
    processed_img = add_gaussian_noise(img, mean, sigma)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'noisy'), class_label)
    element['img'] = np.array(processed_img)
    return element

def split_and_shuffle_grid_element(element, grid_size):
    img = element['img']
    processed_img = split_and_shuffle_grid(img, grid_size)
    class_label = element['label']
    save_image(np.array(processed_img), os.path.join(save_dir, 'shuffled'), class_label)
    element['img'] = np.array(processed_img)
    return element

def setup_pipeline():
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    return pipe

def transform_cifar_class(element, pipe):
    current_class = element['label']
    target_class = class_mapping[current_class]
    source_prompt = prompt_mapping[current_class]
    target_prompt = prompt_mapping[target_class]
    
    init_image = element['img'].resize((768, 768))

    mask_image = pipe.generate_mask(image=init_image, source_prompt=source_prompt, target_prompt=target_prompt)
    image_latents = pipe.invert(image=init_image, prompt=target_prompt).latents
    transformed_image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]
    
    save_image(np.array(transformed_image), os.path.join(save_dir, 'diffedit'), target_class)
    element['img'] = np.array(transformed_image)
    element['label'] = target_class
    return element

def none_process(element):
    img = element['img']
    class_label = element['label']
    save_image(np.array(img), os.path.join(save_dir, 'original'), class_label)
    return element

# Parameters
CROP_SIZE = 16
DOWNSAMPLE_SIZE = 8
MAX_ANGLE = 30  # For a random rotation between -30 and +30 degrees
NOISE_MEAN = 0
NOISE_SIGMA = 25  # Adjust this for more/less noise
GRID_SIZE = (4, 4)  # Define the number of rows and columns for the grid
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

'''
cropped_dataset = dataset['train'].map(random_crop_and_zoom_element, fn_kwargs={'crop_size': CROP_SIZE})
cropped_dataset.save_to_disk("./processed_cifar10/cropped_dataset")

downsampled_dataset = dataset['train'].map(downsample_and_upsample_element, fn_kwargs={'downsample_size': DOWNSAMPLE_SIZE})
downsampled_dataset.save_to_disk("./processed_cifar10/downsampled_dataset")

edged_dataset = dataset['train'].map(abstract_representation_element)
edged_dataset.save_to_disk("./processed_cifar10/edged_dataset")

rotated_dataset = dataset['train'].map(random_rotation_element, fn_kwargs={'max_angle': MAX_ANGLE})
rotated_dataset.save_to_disk("./processed_cifar10/rotated_dataset")

noisy_dataset = dataset['train'].map(add_gaussian_noise_element, fn_kwargs={'mean': NOISE_MEAN, 'sigma': NOISE_SIGMA})
noisy_dataset.save_to_disk("./processed_cifar10/noisy_dataset")

shuffled_dataset = dataset['train'].map(lambda x: split_and_shuffle_grid_element(x, GRID_SIZE))
shuffled_dataset.save_to_disk("./processed_cifar10/shuffled_dataset")
'''
original_dataset = dataset['train'].map(none_process)
original_dataset.save_to_disk("./processed_cifar10/original_dataset")

'''
# Modify dataset
pipe = setup_pipeline()
transformed_dataset = dataset['train'].map(lambda x: transform_cifar_class(x, pipe))
transformed_dataset.save_to_disk("./processed_cifar10/transformed_dataset")

'''