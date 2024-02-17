from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset, load_from_disk
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import os

#dataset = load_from_disk("processed_cifar10/noisy_dataset")
dataset = load_from_disk("processed_cifar10/rotated_dataset")
#model_ckpt = "google/vit-base-patch16-224-in21k"
model_ckpt = "google/vit-large-patch16-224-in21k"
#model_ckpt = "google/vit-huge-patch14-224-in21k"
#model_ckpt = "facebook/dino-vitb16"
#model_ckpt = "thapasushil/vit-base-cifar10"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

labels = dataset["label"]
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

import torchvision.transforms as T


# Data transformation chain.
transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["img"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            outputs = model(**new_batch, output_hidden_states=True)
            #embeddings = {f"embeddings_{i}": torch.mean(output, dim=1).cpu().numpy() for i, output in enumerate(outputs.hidden_states)}
            embeddings = {f"embeddings_{i}": output[:, 0].cpu().numpy() for i, output in enumerate(outputs.hidden_states)}
        return embeddings

    return pp

# Here, we map embedding extraction utility on our subset of candidate images.
batch_size = 24
device = "cuda" if torch.cuda.is_available() else "cpu"
extract_fn = extract_embeddings(model.to(device))
candidate_subset_emb = dataset.map(extract_fn, batched=True, batch_size=24)
candidate_ids = []
for id in tqdm(range(len(candidate_subset_emb))):
    label = candidate_subset_emb[id]["label"]

    # Create a unique indentifier.
    entry = str(id) + "_" + str(label)

    candidate_ids.append(entry)
    
num_layers = len([col for col in candidate_subset_emb.column_names if "embeddings_" in col])
all_candidate_embeddings = {f"embeddings_{i}": candidate_subset_emb[f"embeddings_{i}"] for i in range(num_layers)}
all_candidate_embeddings = {key: torch.from_numpy(np.array(val)) for key, val in all_candidate_embeddings.items()}

def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)

    '''# Normalize embeddings
    emb_one = emb_one / torch.norm(emb_one, dim=1, keepdim=True)
    emb_two = emb_two / torch.norm(emb_two, dim=1, keepdim=True)
    
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)'''

    return scores.numpy().tolist()

def fetch_similar(image, top_k=5):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    image_transformed = transformation_chain(image).unsqueeze(0)
    new_batch = {"pixel_values": image_transformed.to(device)}

    # Compute the embedding.
    with torch.no_grad():
        query_outputs = model(**new_batch, output_hidden_states=True)
        query_embeddings = [torch.from_numpy(output[:, 0].cpu().numpy()) for output in query_outputs.hidden_states]

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    layer_similarities = []
    top_scores = []
    average_scores = []
    average_same_label_scores = []  # This list will store the average score for same label images for each layer
    for layer_num in tqdm(range(len(query_embeddings))):
        sim_scores = compute_scores(all_candidate_embeddings[f"embeddings_{layer_num}"], query_embeddings[layer_num])
        similarity_mapping = dict(zip(candidate_ids, sim_scores))

        average_score = np.mean(sim_scores)
        average_scores.append(average_score)

        # Calculate average score for images with the same label
        same_label_scores = [score for id_score, score in similarity_mapping.items() if int(id_score.split("_")[-1]) == int(test_label)]
        average_same_label_score = np.mean(same_label_scores) if same_label_scores else 0
        average_same_label_scores.append(average_same_label_score)


        # Sort the mapping dictionary and return `top_k` candidates.
        similarity_mapping_sorted = dict(
            sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
        )
        id_entries = list(similarity_mapping_sorted.keys())[:top_k]

        ids = list(map(lambda x: int(x.split("_")[0]), id_entries))
        labels = list(map(lambda x: int(x.split("_")[-1]), id_entries))
        layer_similarities.append((ids, labels))
        top_scores.append(similarity_mapping_sorted[id_entries[0]])

    return layer_similarities, top_scores, average_scores, average_same_label_scores

print(all_candidate_embeddings)
#target = r"/home/jackhe/LayerChoice/GAN_images/500/1/3.png"
target = r"/home/jackhe/LayerChoice/ddpm_images/ddpm_image/ddpm_generated_image_0.png"
test_sample = Image.open(target)
test_label = '1'

layer_similarities, top_scores, average_scores, average_same_label_scores = fetch_similar(test_sample)
path = f"/home/jackhe/LayerChoice/similar_pic"
save_path = os.path.join(path, "DDPM0_noisy_vitlarge")

isExist = os.path.exists(save_path)
if not isExist:
    os.mkdir(save_path)

def plot_images(images, labels, layer_num, save_path):
    if not isinstance(labels, list):
        labels = labels.tolist()

    fig, ax = plt.subplots(figsize=(20, 10))
    columns = 6
    for (i, image) in enumerate(images):
        label_id = int(labels[i])
        sub_ax = plt.subplot(int(len(images) // columns) + 1, columns, i + 1)
        if i == 0:
            sub_ax.set_title("Query Image\n" + "Label: {}".format(id2label[label_id]))
        else:
            sub_ax.set_title(
                "Similar Image # " + str(i) + "\nLabel: {}".format(id2label[label_id])
            )
        plt.imshow(np.array(image).astype("int"))
        plt.axis("off")
    plt.tight_layout()    
    plt.savefig(f'{save_path}/layer_{layer_num}.png')
    plt.close()

for layer_num, (sim_ids, sim_labels) in enumerate(layer_similarities):
    images = []
    labels = []
    images.append(test_sample)
    labels.append(test_label)
    for id, label in zip(sim_ids, sim_labels):
        image_array = np.array(candidate_subset_emb[id]["img"])
        image = Image.fromarray(image_array)
        images.append(image)
        labels.append(label)
    plot_images(images, labels, layer_num, save_path)

'''
isExist = os.path.exists(image_path)
if not isExist:
    os.mkdir(image_path)

for id, label in zip(sim_ids, sim_labels):
    image_array = np.array(candidate_subset_emb[id]["img"])
    image = Image.fromarray(image_array)
    image.save(f'{image_path}/image_{i}.png')
    i += 1 '''
images = []

# Iterate through each layer
for layer_num in range(len(layer_similarities)):
    # Open image and append to list
    img = Image.open(f'{save_path}/layer_{layer_num}.png')
    images.append(img)

# Concatenate images vertically
final_img = Image.fromarray(np.concatenate([np.array(img) for img in images], axis=0))

# Save the final image
final_img.save(f'{save_path}/all_layers.png')

# Plot the graph

# Set up the figure and the first axis
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(list(range(len(top_scores))), top_scores, marker='o', color='blue', label='top_score')
ax1.plot(list(range(len(average_scores))), average_scores, marker='s', color='orange', label='mean_score')
ax1.plot(list(range(len(average_same_label_scores))), average_same_label_scores, marker='s', color='red', label='mean_score_same_label')

ax1.set_xlabel('Layer Number')
ax1.set_ylabel('Score', color='black')
ax1.tick_params('y', colors='black')

# Create a second axis
ax2 = ax1.twinx()
ax2.plot(list(range(len(average_scores))),  [a / b for a, b in zip(top_scores, average_scores)], marker='x', color='green', label='ratio')
ax2.plot(list(range(len(average_same_label_scores))),  [a / b for a, b in zip(top_scores, average_same_label_scores)], marker='x', color='purple', label='ratio_same_label')

ax2.set_ylabel('Ratio', color='green')
ax2.tick_params('y', colors='black')

ax1.legend()
ax2.legend()

plt.title('Change in Top Similarity Score with Layer Number')
plt.grid(True)
plt.show()

fig.savefig(f'{save_path}/scores.png')
