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
import convolution_autoencoder_pyramid_classification_moreiso as cpcm
import convolution_autoencoder_pyramid_classification_noiso as cpcn
import convolution_autoencoder_isotropic_classification as cic

dataset = load_dataset("cifar10")

#model_name = "Inception_V3"
model_name = "isotropic_classification"
generated_data_source = "GAN500"
for generated_data_source in ["GAN500", "GAN10000", "DDPM500"]:
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
    '''
    model = HookedModel(
        timm.create_model(
            #'inception_v3',
            'vgg16',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ),
        
        #layer_names=["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3" , "Pool1", "Conv2d_3b_1x1", "Conv2d_4a_3x3", "Pool2", "Mixed_5b", "Mixed_5b.branch_pool" "Mixed_5c", "Mixed_5c.branch_pool", "Mixed_5d", "Mixed_5d.branch_pool", "Mixed_6a", "Mixed_6b", "Mixed_6b.branch_pool", "Mixed_6c", "Mixed_6c.branch_pool", "Mixed_6d", "Mixed_6d.branch_pool", "Mixed_6e", "Mixed_6e.branch_pool", "Mixed_7a", "Mixed_7b", "Mixed_7b.branch_pool", "Mixed_7c", "Mixed_7c.branch_pool", "global_pool"]
        layer_names = ["features", "features.0", "features.5", "features.10", "features.15", "features.20", "features.25", "features.30", "pre_logits", "head", "head.global_pool", "head.fc"]
    )
    '''

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


    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

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
                avgpool = nn.AvgPool2d(kernel_size=2, stride=4)
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
        # Otherwise, compute the embeddings and save them
        candidate_subset_emb_train = dataset["train"].map(extract_fn, batched=True, batch_size=24)
        print("finish mapping")

        print(candidate_subset_emb_train)

        num_layers = len([col for col in candidate_subset_emb_train.column_names if "embeddings_" in col])
        all_candidate_embeddings_train = {f"embeddings_{i}": candidate_subset_emb_train[f"embeddings_{i}"] for i in tqdm(range(num_layers))}
        all_candidate_embeddings_train = {key: torch.from_numpy(np.array(val)) for key, val in all_candidate_embeddings_train.items()}
        torch.save(all_candidate_embeddings_train, f"embeddings/all_candidate_embeddings_train_{model_name}.pth")
        print(f"all_candidate_embeddings_train_{model_name} SAVED")

    if os.path.isfile(f"embeddings/all_candidate_embeddings_test_{model_name}.pth"):
        all_candidate_embeddings_test = torch.load(f"embeddings/all_candidate_embeddings_test_{model_name}.pth")
        print(f"all_candidate_embeddings_test_{model_name} loaded")
    else:
        candidate_subset_emb_test = dataset["test"].map(extract_fn, batched=True, batch_size=24)
        print("finish mapping")

        num_layers = len([col for col in candidate_subset_emb_test.column_names if "embeddings_" in col])
        all_candidate_embeddings_test = {f"embeddings_{i}": candidate_subset_emb_test[f"embeddings_{i}"] for i in tqdm(range(num_layers))}
        all_candidate_embeddings_test = {key: torch.from_numpy(np.array(val)) for key, val in all_candidate_embeddings_test.items()}

        torch.save(all_candidate_embeddings_test, f"embeddings/all_candidate_embeddings_test_{model_name}.pth")
        print(f"all_candidate_embeddings_test_{model_name} SAVED")


    def plot_for_multiple_paths(layers, all_candidate_embeddings_train, all_candidate_embeddings_test):
        scores = []
        layer_num = []
        for layer_index, layer in enumerate(tqdm(layers)):
        # Convert the lists to numpy arrays
            gen = np.vstack([t.cpu().numpy() for t in layer]).astype('float')
            gen = gen.reshape(gen.shape[0], -1)
            
            embeddings_train_layer = all_candidate_embeddings_train[f"embeddings_{layer_index}"].numpy().astype('float')
            embeddings_test_layer = all_candidate_embeddings_test[f"embeddings_{layer_index}"].numpy().astype('float')
            Pn = embeddings_test_layer  # Use testX as the test sample
            Qm = gen #needs to be generated samples
            T = embeddings_train_layer.astype('float')
            
            print("PCA")
            layer_num.append(layer_index)
            n_components=min(T.shape[0], T.shape[1], 64)
            pca = PCA(n_components)
            print(n_components)
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

            ct = C_T(Pn_projected, Pn_cells, Qm_projected, Qm_cells, T_projected, T_cells, tau = 10 / len(Qm)) #tau is the threshold fraction of samples from Q that must exist in a cell for it to be evaluated by the metric
            
            print(ct)
            scores.append(ct)


        plt.plot(layer_num, scores, marker='o', label=f"{generated_data_source}")
        plt.xlabel("layers")
        plt.ylabel("CT scores")
        plt.title(f"CT vs layer with encoder {model_name}")
        plt.legend()
        plt.savefig(f'{save_path}/{generated_data_source}/ct_v_layer_{model_name}.png')
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
                avgpool = nn.AvgPool2d(kernel_size=2, stride=4)
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