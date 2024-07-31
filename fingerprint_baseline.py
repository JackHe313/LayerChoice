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

def calculate_distance(mean1, std1, fid1, mean2, std2, fid2):
    mean_distance = np.linalg.norm(mean1 - mean2)
    std_distance = np.linalg.norm(std1 - std2)
    if fid1 and fid2:
        fid_distance = np.linalg.norm(fid1 - fid2)
    else:
        fid_distance = 0
    return fid_distance  # Combine distances for simplicity

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1.flatten(), vec2.flatten())
    norm_product = np.linalg.norm(vec1.flatten()) * np.linalg.norm(vec2.flatten())
    return dot_product / norm_product if norm_product else 0

def combined_metric(cosine_sim, l2_dist, alpha):
    """
    Combine cosine similarity and L2 distance into a single metric.
    Alpha determines the weight of the cosine similarity relative to the L2 distance.
    Cosine similarity is given more importance with a higher alpha.
    """
    # Normalize L2 distance to a similarity measure
    l2_similarity = 1 / (1 + l2_dist)
    # Combine the two metrics
    return alpha * cosine_sim + (1 - alpha) * l2_similarity

def getArch(folder):
    name = folder.split('/')[5]
    if len(name) == 5:
        return name[:4]
    return name.split('-')[1]

def compute_accuracy(train_folders, test_folders, stats_file=None, alpha=0.5):
    stats = {}
    if stats_file:
        stats = load_statistics_from_file(stats_file)

    correct_matches = 0
    for i, target_folder in enumerate(tqdm(test_folders)):
        if stats_file:
            target_mean, target_std, target_fid = stats[target_folder]
        else:
            target_images = load_and_preprocess_images(target_folder)
            target_mean, target_std = calculate_statistics(target_images)
            target_fid = None

        distances = []
        for j, folder in enumerate(train_folders):
            if stats_file:
                gen_mean, gen_std, gen_fid= stats[folder]
            else:
                generated_images = load_and_preprocess_images(folder)
                gen_mean, gen_std = calculate_statistics(generated_images)
                gen_fid = None
            distance = calculate_distance(gen_mean, gen_std, gen_fid, target_mean, target_std, target_fid)
            cos_sim = cosine_similarity(gen_mean, target_mean) + cosine_similarity(gen_std, target_std)
            combined = combined_metric(cos_sim, distance, alpha)
            distances.append((combined, getArch(folder)))

        # Sort distances, but keep model arch info
        distances.sort(reverse=True)
        print(f"Target folder: {target_folder}")
        print(f"Closest matches: {distances[:3]}")

        # Check if the closest match has the same model architecture
        if distances and distances[0][1] == getArch(target_folder):
            correct_matches += 1

    total_targets = len(test_folders)
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

def load_fid_scores(fid_score_path):
    fid_scores = {}
    with open(fid_score_path, 'r') as file:
        for line in file:
            path, score = line.strip().split()
            fid_scores[path] = float(score)
    return fid_scores

def load_statistics_from_file(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        stats = data['stats'].item()  # Assume stats are stored in a dictionary format
    fid_scores = load_fid_scores("/home/jackhe/LayerChoice/fid_scores/fid_score.txt")
    # fid_scores = load_fid_scores("/home/jackhe/LayerChoice/fid_scores/fid_score_small.txt")
    for folder in stats.keys():
        # Update the stats dictionary with the FID score if available
        if folder in fid_scores:
            stats[folder] = stats[folder] + (fid_scores[folder],)
        else:
            # Assign a default value or handle missing FID score
            stats[folder] = stats[folder] + (None,)
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
    
    # train_folders = ['/home/jackhe/LayerChoice/ddpm_images/ddim2',
    #             '/home/jackhe/LayerChoice/ddpm_images/ddpm1',
    #             '/home/jackhe/LayerChoice/ddpm_images/pndm2',
    #             '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_50_43/fake',
    #             '/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_01_24_34/fake',
    #             '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_21_44/fake',]
    
    test_folders = ['/home/jackhe/LayerChoice/samples/CIFAR10-ContraGAN-train-2022_01_13_11_17_17/fake',
                    '/home/jackhe/LayerChoice/samples/CIFAR10-BigGAN-Deep-train-2022_02_02_21_48_16/fake',
                    '/home/jackhe/LayerChoice/samples/CIFAR10-SNGAN-train-2022_03_06_02_23_48/fake',
                    '/home/jackhe/LayerChoice/ddpm_images/ddpm2',
                    '/home/jackhe/LayerChoice/ddpm_images/pndm1',
                    '/home/jackhe/LayerChoice/ddpm_images/ddim1']

    target_folder = '/home/jackhe/LayerChoice/ddpm_images/ddpm2'
    
    args = parser.parse_args()

    stats_file = args.load

    if args.accuracy:
        compute_accuracy(train_folders, test_folders, stats_file=stats_file)
    elif args.save:
        import time
        start_time = time.time()
        save_statistics_to_file(train_folders, args.save)
        print(f"Time taken: {time.time() - start_time:.2f} seconds" )
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
            
            distance = calculate_distance(gen_mean, gen_std, None, target_mean, target_std, None)

            gen_features = np.concatenate((gen_mean.flatten(), gen_std.flatten()))
            target_features = np.concatenate((target_mean.flatten(), target_std.flatten()))
            cos_sim = cosine_similarity(gen_features, target_features)
            combined = combined_metric(cos_sim, distance, alpha=0.5)
            distances.append(combined)

        # Determine the best fit
        closest_index = np.argmax(distances)
        print(f"The target folder is: {target_folder}")
        print(f"The best fit is from folder: {generated_folders[closest_index]}")
