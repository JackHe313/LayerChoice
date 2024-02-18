import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm

def load_and_preprocess_images(folder_path, size=(256, 256)):
    images = []
    for dir in os.listdir(folder_path):
        dirpath = os.path.join(folder_path, dir)
        for filename in os.listdir(dirpath):
            if filename.endswith('.png'):  
                img = Image.open(os.path.join(dirpath, filename))
                img = img.resize(size)  # Resize to common size
                images.append(np.asarray(img, dtype=np.float32))
    return np.stack(images)

def calculate_statistics(images):
    mean_image = np.mean(images, axis=0)
    std_dev_image = np.std(images, axis=0)
    return mean_image, std_dev_image

def calculate_distance(mean1, std1, mean2, std2):
    # Simple Euclidean distance for demonstration; you might need a more sophisticated measure
    mean_distance = np.linalg.norm(mean1 - mean2)
    std_distance = np.linalg.norm(std1 - std2)
    return mean_distance + std_distance  # Combine distances for simplicity

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    norm_product = np.linalg.norm(vec1.flatten()) * np.linalg.norm(vec2.flatten())
    return dot_product / norm_product if norm_product else 0

def combined_metric(cosine_sim, l2_dist, alpha=0.5):
    """
    Combine cosine similarity and L2 distance into a single metric.
    Alpha determines the weight of the cosine similarity relative to the L2 distance.
    Cosine similarity is given more importance with a higher alpha.
    """
    # Normalize L2 distance to a similarity measure
    l2_similarity = 1 / (1 + l2_dist)
    # Combine the two metrics
    return alpha * cosine_sim + (1 - alpha) * l2_similarity

def compute_accuracy(generated_folders, modelArch, stats_file=None, alpha=0.1):
    stats = {}
    if stats_file:
        stats = load_statistics_from_file(stats_file)

    correct_matches = 0
    for i, target_folder in enumerate(tqdm(generated_folders)):
        if stats_file:
            target_mean, target_std = stats[target_folder]
        else:
            target_images = load_and_preprocess_images(target_folder)
            target_mean, target_std = calculate_statistics(target_images)

        distances = []
        for j, folder in enumerate(generated_folders):
            if i != j:  # Skip the target folder
                if stats_file:
                    gen_mean, gen_std = stats[folder]
                else:
                    generated_images = load_and_preprocess_images(folder)
                    gen_mean, gen_std = calculate_statistics(generated_images)
                distance = calculate_distance(gen_mean, gen_std, target_mean, target_std)
                cos_sim = cosine_similarity(gen_mean, target_mean) + cosine_similarity(gen_std, target_std)
                combined = combined_metric(cos_sim, distance, alpha)
                distances.append((combined, modelArch[j]))

        # Sort distances, but keep model arch info
        distances.sort(reverse=True)

        # Check if the closest match has the same model architecture
        if distances and distances[0][1] == modelArch[i]:
            correct_matches += 1

    total_targets = len(modelArch)
    accuracy = correct_matches / total_targets
    print(f"Accuracy: {accuracy}")

def save_statistics_to_file(generated_folders, file_path):
    stats = {}
    for folder in tqdm(generated_folders):
        images = load_and_preprocess_images(folder)
        mean, std = calculate_statistics(images)
        # Use folder name as the key
        stats[folder] = (mean, std)
    
    # Save the dictionary to a file
    with open(file_path, 'wb') as f:
        np.savez(f, stats=stats)

    print(f"Statistics saved to {file_path}")

def load_statistics_from_file(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        stats = data['stats'].item()  # Assume stats are stored in a dictionary format
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare image sets and compute model accuracy.")
    parser.add_argument("--accuracy", "-a", action='store_true', help="Flag to compute model accuracy")
    parser.add_argument("--save", '-s', type=str, help="Path to save statistics")
    parser.add_argument("--load", '-l', type=str, help="Path to load statistics")

    # Paths to your image folders
    generated_folders = ['/home/jackhe/LayerChoice/ddpm_images/ddim1', 
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
                        '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_24_46/fake']

    modelArch = ['ddim','ddim','ddim',
                'ddpm','ddpm','ddpm',
                'pndm','pndm','pndm',
                'BigGAN', 'BigGAN', 'BigGAN',
                'ContraGAN', 'ContraGAN', 'ContraGAN',
                'SNGAN', 'SNGAN', 'SNGAN']

    target_folder = '/home/jackhe/LayerChoice/ddpm_images/ddpm2'
    
    args = parser.parse_args()

    stats_file = args.load

    if args.accuracy:
        compute_accuracy(generated_folders, modelArch, stats_file=stats_file)
    elif args.save:
        save_statistics_to_file(generated_folders, args.save)
    else:
        stats = {}
        if stats_file:
            stats = load_statistics_from_file(stats_file)

        # Calculate statistics for the target set
        target_images = load_and_preprocess_images(target_folder)
        target_mean, target_std = calculate_statistics(target_images)

        distances = []
        for j, folder in enumerate(tqdm(generated_folders)):
            if stats_file:
                gen_mean, gen_std = stats[folder]
            else:
                generated_images = load_and_preprocess_images(folder)
                gen_mean, gen_std = calculate_statistics(generated_images)
            
            distance = calculate_distance(gen_mean, gen_std, target_mean, target_std)

            gen_features = np.concatenate((gen_mean.flatten(), gen_std.flatten()))
            target_features = np.concatenate((target_mean.flatten(), target_std.flatten()))
            cos_sim = cosine_similarity(gen_features, target_features)
            combined = combined_metric(cos_sim, distance, alpha=0.5)
            distances.append(combined)

        # Determine the best fit
        closest_index = np.argmax(distances)
        print(f"The target folder is: {target_folder}")
        print(f"The best fit is from folder: {generated_folders[closest_index]}")
