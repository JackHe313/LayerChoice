import numpy as np
from data_copying_tests import C_T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import os
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse

dataset = load_dataset("cifar10")

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
model_name = "mutual_information_autoencoder"
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder layers
        self.Encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(13, 16, 3, padding=1)),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            
            ('conv2', nn.Conv2d(16, 16, 3, padding=1)),
            ('bn2', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU()),
            
            ('conv3', nn.Conv2d(16, 32, 3, padding=1, stride=2)),
            ('bn3', nn.BatchNorm2d(32)),
            ('relu3', nn.ReLU()),
            
            ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
            ('bn4', nn.BatchNorm2d(32)),
            ('relu4', nn.ReLU()),
            
            ('conv5', nn.Conv2d(32, 64, 3, padding=1, stride=2)),
            ('bn5', nn.BatchNorm2d(64)),
            ('relu5', nn.ReLU()),
            
            ('conv6', nn.Conv2d(64, 64, 3, padding=1)),
            ('bn6', nn.BatchNorm2d(64)),
            ('relu6', nn.ReLU())
        ]))
        
        # Decoder layers
        self.Decoder = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(64, 64, 3, padding=1)),
            ('d_relu1', nn.ReLU()),
            
            ('deconv2', nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1)),
            ('d_relu2', nn.ReLU()),
            
            ('deconv3', nn.ConvTranspose2d(32, 32, 3, padding=1)),
            ('d_relu3', nn.ReLU()),
            
            ('deconv4', nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=1)),
            ('d_relu4', nn.ReLU()),
            
            ('deconv5', nn.ConvTranspose2d(16, 16, 3, padding=1)),
            ('d_relu5', nn.ReLU()),
            
            ('deconv6', nn.ConvTranspose2d(16, 3, 3, padding=1)),
            ('output_tanh', nn.Tanh())
        ]))

    def forward(self, x):
        encoded_x = self.Encoder(x)
        decoded_x = self.Decoder(encoded_x)
        return decoded_x

if __name__ == "__main__":
    ckpt_list = ["google/vit-base-patch16-224-in21k", "google/vit-large-patch16-224-in21k", "google/vit-huge-patch14-224-in21k", "thapasushil/vit-base-cifar10", "facebook/dino-vitb16"]
    for model_ckpt in tqdm(ckpt_list):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_name = model_ckpt.split('/')[-1]
        data_model = AutoModel.from_pretrained(model_ckpt).to(device)
        print(data_name)
        
        # Here, we map embedding extraction utility on our subset of candidate images.
        batch_size = 24

        path = f"/home/jackhe/PyTorch-StudioGAN/"
        save_path = os.path.join(path, "Mutual_Information")

        transform = transforms.Compose([
            transforms.Resize(int((256 / 224) * 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_target_data = dataset["train"]['img']
        test_target_data = dataset["test"]['img']

        num_workers = 0
        # how many samples per batch to load
        batch_size = 20

        # initialize the NN
        model = ConvAutoencoder().to(device)

        # specify loss function
        criterion = nn.MSELoss()

        # specify loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start_epoch = 1

        # number of epochs to train the model
        parser = argparse.ArgumentParser(description='mutual information Autoencoder Training')
        parser.add_argument('-e','--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
        args = parser.parse_args()

        n_epochs = args.epochs

        print(f"{model_name}_{n_epochs}")

        train_losses = []  # List to record training losses for each epoch
        test_losses = []   # List to record test losses for each epoch

        if os.path.isfile(f"autoencoders/{model_name}/autoencoder_model_{data_name}_{n_epochs}.pth"):
            checkpoint = torch.load(f'autoencoders/{model_name}/autoencoder_model_{data_name}_{n_epochs}.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            loss = checkpoint['loss']
            model.train()  # Ensure the model is in training mode
            print(f'{model_name}_autoencoder_model_{data_name}_{n_epochs}.pth loaded')

        for epoch in tqdm(range(start_epoch, n_epochs+1)):
            # monitor training loss
            train_loss = 0.0
            
            ###################
            # train the model #
            ###################
            for images in train_target_data:
                image_transformed = transform(images).unsqueeze(0).to(device)
                new_batch = {"pixel_values": image_transformed.to(device)}
                # Compute the embedding.
                with torch.no_grad():
                    query_outputs = data_model(**new_batch, output_hidden_states=True)
                    result = [output[:, 0].view(output.size(0), -1) for output in query_outputs.hidden_states]
                embedding = torch.stack(result, dim=0).to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(embedding.unsqueeze(0))
                print(outputs.shape)
                # calculate the loss
                loss = criterion(outputs, image_transformed)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item()*embedding.size(0)
                    
            # print avg training statistics 
            train_loss = train_loss/len(train_target_data)
            train_losses.append(train_loss)

            # Compute test loss
            model.eval()  # Set the model to evaluation mode
            test_loss = 0.0

            with torch.no_grad():
                for images in test_target_data:
                    image_transformed = transform(images).unsqueeze(0).to(device)
                    new_batch = {"pixel_values": image_transformed.to(device)}
                    
                    query_outputs = data_model(**new_batch, output_hidden_states=True)
                    result = [output[:, 0].view(output.size(0), -1) for output in query_outputs.hidden_states]
                    embedding = torch.stack(result, dim=0).to(device)
                    embedding = embedding.to(device)
                    outputs = model(embedding.unsqueeze(0))
                    loss = criterion(outputs, images)
                    test_loss += loss.item()*embedding.size(0)

            test_loss = test_loss/len(test_target_data)
            test_losses.append(test_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss, test_loss))

            model.train()  # Set the model back to training mode

        # After all epochs, plot the training and test loss curves
        plt.figure(figsize=(10, 7))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.title("Training and Test Losses over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"autoencoders/{model_name}/{data_name}_losses_over_epochs.png")
        plt.close()

        print(test_losses)
'''
        if(start_epoch < n_epochs):
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'train_losses': train_losses,
                    'test_losses':test_losses
                }, f'{model_name}_autoencoder_model_{data_name}_{n_epochs}.pth')

        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # helper function to un-normalize and display an image
        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images = images.to(device)
        output = model(images)
        images = images.cpu().numpy()
        output = output.view(batch_size, 3, 32, 32).detach().cpu().numpy()
        # Plot original images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[labels[idx]])
        plt.savefig("original_images.png")
        plt.close()  # Close the plot to free up resources

        # Plot reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
            imshow(output[idx])
            ax.set_title(classes[labels[idx]])
        plt.savefig(f"autoencoders/{model_name}/reconstructed_images_{n_epochs}.png")
        plt.close()  # Close the plot to free up resources

        print("plot saved")

'''

