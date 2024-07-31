import os
import random
import shutil

# Define the paths and labels
generated_folders = [
    '/home/jackhe/LayerChoice/ddpm_images/ddim1', 
    '/home/jackhe/LayerChoice/ddpm_images/ddim2',
    '/home/jackhe/LayerChoice/ddpm_images/ddim3',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm1',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm2',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm3',
    '/home/jackhe/LayerChoice/ddpm_images/pndm1',
    '/home/jackhe/LayerChoice/ddpm_images/pndm2',
    '/home/jackhe/LayerChoice/ddpm_images/pndm3',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_48_16/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_50_43/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_14_18_23_41/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_24_34/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_25_00/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_11_17_17/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_21_44/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_23_48/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_24_46/fake'
]

modelArch = [
    'ddim', 'ddim', 'ddim',
    'ddpm', 'ddpm', 'ddpm',
    'pndm','pndm','pndm',
    'BigGAN', 'BigGAN', 'BigGAN',
    'ContraGAN', 'ContraGAN', 'ContraGAN',
    'SNGAN', 'SNGAN', 'SNGAN'
]

# Create directories for training and testing datasets
base_dir = '/home/jackhe/LayerChoice/fingerprinting_dataset_small' # full: 900 train 600 test   //  medium: 600 train 600 test  // small: 150 train 600 test
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for each model architecture
for arch in set(modelArch):
    os.makedirs(os.path.join(train_dir, arch), exist_ok=True)
    os.makedirs(os.path.join(test_dir, arch), exist_ok=True)

# Counters for unique naming
train_counter = 0
test_counter = 0

# Split the dataset
for folder, arch in zip(generated_folders, modelArch):
    all_images = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            images = os.listdir(subfolder_path)
            all_images.extend([os.path.join(subfolder_path, img) for img in images if os.path.isfile(os.path.join(subfolder_path, img))])
    
    print(f"Model: {arch}, Total images: {len(all_images)}")
    train_images = all_images[:50]
    test_images = all_images[300:]  
    print(f"Train images: {len(train_images)}, Test images: {len(test_images)}")

    for img in train_images:
        if os.path.isfile(img):
            unique_name = f"{train_counter}_{os.path.basename(img)}"
            dst_path = os.path.join(train_dir, arch, unique_name)
            shutil.copy(img, dst_path)
            train_counter += 1
        else:
            print(f"Error: File {img} does not exist.")
    
    for img in test_images:
        if os.path.isfile(img):
            unique_name = f"{test_counter}_{os.path.basename(img)}"
            dst_path = os.path.join(test_dir, arch, unique_name)
            shutil.copy(img, dst_path)
            test_counter += 1
        else:
            print(f"Error: File {img} does not exist.")

print("Dataset preparation complete.")

