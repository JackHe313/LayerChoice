import numpy as np
from data_copying_tests import C_T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from PIL import Image
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import OrderedDict
import torchvision.transforms as T
import timm
import convolution_autoencoder_pyramid_reconstruction as cpr
import convolution_autoencoder_isotropic_reconstruction as cir
import convolution_autoencoder_pyramid_classification as cpc
import convolution_autoencoder_isotropic_classification as cic

dataset = load_dataset("cifar10")

#model_name = "Inception_V3"
model_name = "isotropic_classification"
generated_data_source = "GAN500"

if generated_data_source == "GAN500":
    target_paths = r"/home/jackhe/LayerChoice/GAN_images/500"
elif generated_data_source == "GAN10000":
    target_paths = r"/home/jackhe/LayerChoice/GAN_images/10000"
elif generated_data_source == "DDPM500":
    target_paths = r"/home/jackhe/LayerChoice/ddpm_images"
else:
    print("Unknown generated_data_source!")
    target_paths = None

print(target_paths)

class HookedModel(torch.nn.Module):
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.outputs = []
        self.jump = 0
        self.layer_names = layer_names

        for name, layer in self.model.named_modules():
            if 'conv' in name or 'fc' in name :
                layer.register_forward_hook(self.hook)
                print(name)

    def hook(self, module, input, output):
        self.outputs.append(output)

    def forward(self, x):
        self.outputs = []  # Clear the output list
        return self.model(x)

model = HookedModel(
    model = cic.ClassificationModel(10),
    layer_names = []
)

if os.path.isfile(f"autoencoders/{model_name}/autoencoder_model_200.pth"):
    checkpoint = torch.load(f"autoencoders/{model_name}/autoencoder_model_200.pth")
    new_state_dict = OrderedDict()
    for old_key, new_key in zip(checkpoint['model_state_dict'].keys(), model.model.state_dict().keys()):
        new_state_dict[new_key] = checkpoint['model_state_dict'][old_key]
    model.model.load_state_dict(new_state_dict)
    print(f"autoencoders/{model_name}/autoencoder_model_200.pth loaded")

#for name, layer in model.named_modules():
#    print(name)
model = model.eval()

def get_device(model):
    return next(model.parameters()).device

# get model specific transforms (normalization, resize)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = get_device(model)

    def pp(batch):
        images = batch["img"]
        image_batch_transformed = torch.stack(
            [transform(image) for image in images]
        )

        # Move the input data to the GPU
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            model(new_batch["pixel_values"])
            embeddings = {f"embeddings_{i}": torch.mean(output, dim=1).view(output.size(0), -1).cpu().numpy() for i, output in enumerate(model.outputs)}
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
    print("no train embeddings found")

if os.path.isfile(f"embeddings/all_candidate_embeddings_test_{model_name}.pth"):
    all_candidate_embeddings_test = torch.load(f"embeddings/all_candidate_embeddings_test_{model_name}.pth")
    print(f"all_candidate_embeddings_test_{model_name} loaded")
else:
    print("no test embeddings found")


def plot_for_multiple_paths(layers, all_candidate_embeddings_train, all_candidate_embeddings_test):
    scores = []
    layer_num = []
    layer_index = 3
    gen = np.vstack([t for t in layers[layer_index]]).astype('float')
    embeddings_train_layer = all_candidate_embeddings_train[f"embeddings_{layer_index}"]
    embeddings_test_layer = all_candidate_embeddings_test[f"embeddings_{layer_index}"]

    avgpool = nn.AvgPool2d(kernel_size=3, padding=1)
    gen_pooled = torch.tensor(gen).view(-1, 1024, 1, 1)
    train_pooled = embeddings_train_layer.view(-1, 1024, 1, 1)
    test_pooled = embeddings_test_layer.view(-1, 1024, 1, 1)

    for index, layer in enumerate(tqdm(layers)):
    # took the first layer
        Pn = test_pooled.view(-1, 1024).cpu().numpy().astype('float')  # Use testX as the test sample
        Qm = gen_pooled.view(-1, 1024).cpu().numpy().astype('float')
        T = train_pooled.view(-1, 1024).cpu().numpy().astype('float')
        layer_num.append(index)
        pca = PCA(64)
        pca.fit(T)
        Pn_projected = pca.transform(Pn)
        Qm_projected = pca.transform(Qm)
        T_projected = pca.transform(T)

    #get instance space partition
        print('Done PCA, Getting instance space partition...')
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

        ct = C_T(Pn_projected, Pn_cells, Qm_projected, Qm_cells, T_projected, T_cells, tau = 10 / len(Qm)) #tau is the threshold fraction of samples from Q that must exist in a cell for it to be evaluated by the metric
        
        print(ct)
        scores.append(ct)

        gen_pooled = avgpool(gen_pooled)
        train_pooled = avgpool(train_pooled)
        test_pooled = avgpool(test_pooled)
        


    plt.plot(layer_num, scores, marker='o', label=f"{generated_data_source}")
    plt.xticks(layer_num)
    plt.xlabel("pca components")
    plt.ylabel("CT scores")
    plt.title(f"CT vs layer with encoder {model_name}")
    plt.legend()
    plt.savefig(f'{save_path}/{generated_data_source}/ct_avgpooling_{model_name}:L{layer_index}.png')
    return scores

path = f"/home/jackhe/LayerChoice/"
save_path = os.path.join(path, "CT_score")

query_embeddings = []
for dir in os.listdir(target_paths):
    dirpath = os.path.join(target_paths, dir)
    for filename in os.listdir(dirpath):
        target = os.path.join(dirpath, filename)
        test_sample = Image.open(target)
    
        image_transformed = transform(test_sample).unsqueeze(0)
        new_batch = {"pixel_values": image_transformed.to(device)}
        # Compute the embedding.
        with torch.no_grad():
            model(new_batch["pixel_values"])
            result = [torch.mean(output, dim=1).view(output.size(0), -1).cpu() for output in model.outputs]
            query_embeddings.append(result)
layers = []
for i in range(len(query_embeddings[0])):
    layers.append([])
for result in query_embeddings:
    for layer_index, output in enumerate(result):
        layers[layer_index].append(output)
scores = plot_for_multiple_paths(layers, all_candidate_embeddings_train, all_candidate_embeddings_test)
print("CT scores: ", scores)