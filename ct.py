import numpy as np
from data_copying_tests import C_T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import torchvision.transforms as T
import argparse

dataset = load_dataset("cifar10")

parser = argparse.ArgumentParser(description='Process target paths for generated data.')
parser.add_argument('generated_data_source', type=str, help='The source of the generated data')
parser.add_argument('--custom_path','-c', type=str, help='Custom path for the data. If provided, it overrides the default path.')
parser.add_argument('--save', '-s', action='store_true', help='Save the model name and CT-score to a file')
args = parser.parse_args()

# Mapping of data sources to their default paths
default_paths = {
    "GAN500": r"/home/jackhe/LayerChoice/GAN_images/500",
    "GAN10000": r"/home/jackhe/LayerChoice/GAN_images/10000",
    "DDPM": r"/home/jackhe/LayerChoice/ddpm_images/ddpm",
    "rotated": r"/home/jackhe/LayerChoice/processed_cifar10/rotated",
    "noisy": r"/home/jackhe/LayerChoice/processed_cifar10/noisy",
    "downsampled": r"/home/jackhe/LayerChoice/processed_cifar10/downsampled",
    "segmented_realworld": r"/home/jackhe/LayerChoice/processed_cifar10/segmented_realworld"
}

if args.custom_path:
    target_paths = args.custom_path
else:
    target_paths = default_paths.get(args.generated_data_source, None)

if target_paths is None:
    print("Unknown generated_data_source! Please provide a valid source or a custom path.")
else:
    print(target_paths)

generated_data_source = args.generated_data_source
path = f"/home/jackhe/LayerChoice/"
save_path = os.path.join(path, "CT_score",f"{generated_data_source}")
os.makedirs(save_path, exist_ok=True)

#model_ckpt = "microsoft/swin-tiny-patch4-window7-224"
#model_ckpt = "microsoft/swin-base-patch4-window7-224-in22k"
#model_ckpt = "microsoft/swin-large-patch4-window7-224-in22k"
#model_ckpt = "microsoft/resnet-50"

#model_ckpt = "facebook/regnet-y-040"

#model_ckpt = "facebook/convnext-large-224-22k-1k"
#model_ckpt = "facebook/convnext-base-224-22k-1k"
#model_ckpt = "facebook/convnext-xlarge-224-22k-1k"

#model_ckpt = "google/vit-base-patch16-224-in21k"
#model_ckpt = "google/vit-large-patch16-224-in21k"
#model_ckpt = "google/vit-huge-patch14-224-in21k"
#model_ckpt = "facebook/dino-vitb16"
#model_ckpt = "thapasushil/vit-base-cifar10"

#model_ckpt = "Zetatech/pvt-tiny-224"
ckpt_list = ["google/vit-base-patch16-224-in21k"]
for model_ckpt in tqdm(ckpt_list):
    model_name = model_ckpt.split('/')[-1]
    print(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    try:
        height = extractor.size["height"]
    except KeyError:
        height = 224
        print("KeyError: use size 224")

    # Data transformation chain.
    transformation_chain = T.Compose(
        [
            T.Resize(int((256 / 224) * height)),
            T.CenterCrop(height),
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
                embeddings = {f"embeddings_{i}": output[:, 0].view(output.size(0), -1).cpu().numpy() for i, output in enumerate(outputs.hidden_states)}
            return embeddings

        return pp

    # Here, we map embedding extraction utility on our subset of candidate images.
    batch_size = 24
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device))

    # Check if the file exists before loading
    if os.path.isfile(f"embeddings/all_candidate_embeddings_train_{model_name}.pth"):
        # Load the file
        all_candidate_embeddings_train = torch.load(f"embeddings/all_candidate_embeddings_train_{model_name}.pth")
        print(f"all_candidate_embeddings_train_{model_name} loaded")
    else:
        # Otherwise, compute the embeddings and save them
        candidate_subset_emb_train = dataset["train"].map(extract_fn, batched=True, batch_size=24)
        candidate_ids = []
        for id in tqdm(range(len(candidate_subset_emb_train))):
            label = candidate_subset_emb_train[id]["label"]

            # Create a unique indentifier.
            entry = str(id) + "_" + str(label)

            candidate_ids.append(entry)
        
        num_layers = len([col for col in candidate_subset_emb_train.column_names if "embeddings_" in col])
        all_candidate_embeddings_train = {f"embeddings_{i}": candidate_subset_emb_train[f"embeddings_{i}"] for i in range(num_layers)}
        all_candidate_embeddings_train = {key: torch.from_numpy(np.array(val)) for key, val in all_candidate_embeddings_train.items()}
        torch.save(all_candidate_embeddings_train, f"embeddings/all_candidate_embeddings_train_{model_name}.pth")

    if os.path.isfile(f"embeddings/all_candidate_embeddings_test_{model_name}.pth"):
        all_candidate_embeddings_test = torch.load(f"embeddings/all_candidate_embeddings_test_{model_name}.pth")
        print(f"all_candidate_embeddings_test_{model_name} loaded")
    else:
        candidate_subset_emb_test = dataset["test"].map(extract_fn, batched=True, batch_size=24)
        candidate_ids = []
        for id in tqdm(range(len(candidate_subset_emb_test))):
            label = candidate_subset_emb_test[id]["label"]

            # Create a unique indentifier.
            entry = str(id) + "_" + str(label)

            candidate_ids.append(entry)
            
        num_layers = len([col for col in candidate_subset_emb_test.column_names if "embeddings_" in col])
        all_candidate_embeddings_test = {f"embeddings_{i}": candidate_subset_emb_test[f"embeddings_{i}"] for i in range(num_layers)}
        all_candidate_embeddings_test = {key: torch.from_numpy(np.array(val)) for key, val in all_candidate_embeddings_test.items()}
        torch.save(all_candidate_embeddings_test, f"embeddings/all_candidate_embeddings_test_{model_name}.pth")


    def plot_for_multiple_paths(layers, all_candidate_embeddings_train, all_candidate_embeddings_test):
        scores = []
        layer_num = []
        for layer_index, layer in enumerate(tqdm(layers)):
            layer_num.append(layer_index)
        # Convert the lists to numpy arrays
            gen = np.vstack([t.cpu().numpy() for t in layer]).astype('float')
            gen = gen.reshape(gen.shape[0], -1)
            embeddings_train_layer = all_candidate_embeddings_train[f"embeddings_{layer_index}"].numpy().astype('float')
            embeddings_test_layer = all_candidate_embeddings_test[f"embeddings_{layer_index}"].numpy().astype('float')
            Pn = embeddings_test_layer  # Use testX as the test sample
            Qm = gen #needs to be generated samples
            T = embeddings_train_layer.astype('float')

        #convert to 64 embeddings
            n_components=min(T.shape[0], T.shape[1],64)
            pca = PCA(n_components)
            pca.fit(T)
            Pn_projected = pca.transform(Pn)
            Qm_projected = pca.transform(Qm)
            T_projected = pca.transform(T)

        #get instance space partition
            print('Getting instance space partition...')
            n_clusters = 10 # of cells
            KM_clf = KMeans(n_clusters, n_init=10).fit(T_projected)
            Pn_cells = KM_clf.predict(Pn_projected)
            T_cells = KM_clf.predict(T_projected)
            Qm_cells = KM_clf.predict(Qm_projected)#duplicate cell labels are allowed becuase they simply divide up the data
            print("# of labels for P: ", len(Pn_cells), "; Q: ", len(Qm_cells), "; T: ", len(T_cells))
            print("size of data for P: ", len(Pn_projected), "; Q: ", len(Qm_projected), "; T: ", len(T_projected))

            # Generate unique cell labels for each partition
            train_cells = np.arange(embeddings_train_layer.shape[0])  # Cell labels for trainX: 0 to (trainX.shape[0] - 1)
            test_cells = np.arange(embeddings_test_layer.shape[0])  # Cell labels for testX: 0 to (testX.shape[0] - 1)'''

        #Pn_cells = test_cells  # Cell labels for testX

            ct = C_T(Pn_projected, Pn_cells, Qm_projected, Qm_cells, T_projected, T_cells, tau = 20 / len(Qm)) #tau is the threshold fraction of samples from Q that must exist in a cell for it to be evaluated by the metric
            
            print(ct)
            scores.append(ct)


        plt.plot(layer_num, scores, marker='o')
        plt.xlabel("layers")
        plt.ylabel("CT scores")
        plt.title(f"CT vs layer with encoder {model_name}")
        plt.savefig(f'{save_path}/ct_v_layer_{model_name}.png')
        plt.close()  # Close the current figure

        return scores

    print(all_candidate_embeddings_train)
    
    query_embeddings = []
    for dir in os.listdir(target_paths):
        dirpath = os.path.join(target_paths, dir)
        for filename in os.listdir(dirpath):
            target = os.path.join(dirpath, filename)
            test_sample = Image.open(target)
            image_transformed = transformation_chain(test_sample).unsqueeze(0)
            new_batch = {"pixel_values": image_transformed.to(device)}
            # Compute the embedding.
            with torch.no_grad():
                query_outputs = model(**new_batch, output_hidden_states=True)
                result = [torch.from_numpy(output[:, 0].view(output.size(0), -1).cpu().numpy()) for output in query_outputs.hidden_states]
                query_embeddings.append(result)
    layers = []
    for i in range(len(query_embeddings[0])):
        layers.append([])
    for result in query_embeddings:
        for layer_index, output in enumerate(result):
            layers[layer_index].append(output)
    scores = plot_for_multiple_paths(layers, all_candidate_embeddings_train, all_candidate_embeddings_test)
    if args.save:
        with open("model_ct_scores.txt", "a") as f:
            f.write(f"{generated_data_source} {scores[1:]}\n")
    print("CT scores ", scores)