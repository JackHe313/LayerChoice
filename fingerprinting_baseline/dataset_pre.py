import os
import shutil

# Define the paths and labels

# train_folders = [
#     '/home/jackhe/LayerChoice/ddpm_images/ddim2',
#     '/home/jackhe/LayerChoice/ddpm_images/ddpm1',
#     '/home/jackhe/LayerChoice/ddpm_images/pndm2',
#     '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_50_43/fake',
#     '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_24_34/fake',
#     '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_21_44/fake'
# ]

train_folders = ['/home/jackhe/LayerChoice/ddpm_images/ddim2',
                '/home/jackhe/LayerChoice/ddpm_images/ddim3',
                '/home/jackhe/LayerChoice/ddpm_images/ddpm1',
                '/home/jackhe/LayerChoice/ddpm_images/ddpm3',
                '/home/jackhe/LayerChoice/ddpm_images/pndm2',
                '/home/jackhe/LayerChoice/ddpm_images/pndm3',
                '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_50_43/fake',
                '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_14_18_23_41/fake',
                '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_24_34/fake',
                '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_25_00/fake',
                '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_21_44/fake',
                '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_24_46/fake']

test_folders = [
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_11_17_17/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_48_16/fake',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_23_48/fake',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm2',
    '/home/jackhe/LayerChoice/ddpm_images/pndm1',
    '/home/jackhe/LayerChoice/ddpm_images/ddim1'
]

modelArch = {
    '/home/jackhe/LayerChoice/ddpm_images/ddim2': 'ddim',
    '/home/jackhe/LayerChoice/ddpm_images/ddim3': 'ddim',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm1': 'ddpm',
    '/home/jackhe/LayerChoice/ddpm_images/pndm2': 'pndm',
    '/home/jackhe/LayerChoice/ddpm_images/pndm3': 'pndm',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_50_43/fake': 'BigGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_14_18_23_41/fake': 'BigGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_24_34/fake': 'ContraGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_25_00/fake': 'ContraGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_21_44/fake': 'SNGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_11_17_17/fake': 'ContraGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_48_16/fake': 'BigGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_23_48/fake': 'SNGAN',
    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_24_46/fake': 'SNGAN',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm2': 'ddpm',
    '/home/jackhe/LayerChoice/ddpm_images/ddpm3': 'ddpm',
    '/home/jackhe/LayerChoice/ddpm_images/pndm1': 'pndm',
    '/home/jackhe/LayerChoice/ddpm_images/ddim1': 'ddim',
}

# Create directories for training and testing datasets
base_dir = '/home/jackhe/LayerChoice/fingerprinting_dataset_full'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for each model architecture
for arch in set(modelArch.values()):
    os.makedirs(os.path.join(train_dir, arch), exist_ok=True)
    os.makedirs(os.path.join(test_dir, arch), exist_ok=True)

# Function to process folders
def process_folders(folders, destination_dir):
    counter = 0
    for folder in folders:
        arch = modelArch[folder]
        print(f"Processing {folder} for {arch}")
        for subfolder in os.listdir(folder):
            for img in os.listdir(os.path.join(folder, subfolder)):
                img_path = os.path.join(folder, subfolder, img)
                if os.path.isfile(img_path):
                    unique_name = f"{counter}_{img}"
                    dst_path = os.path.join(destination_dir, arch, unique_name)
                    shutil.copy(img_path, dst_path)
                    counter += 1
                else:
                    print(f"Error: File {img_path} does not exist.")

# Process training and testing datasets
process_folders(train_folders, train_dir)
process_folders(test_folders, test_dir)

print("Dataset preparation complete.")
