import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import os
from tqdm import tqdm

# Define transformations for the training and testing datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Model, loss function, and optimizer defined.")
base_dir = '/home/jackhe/LayerChoice/fingerprinting_dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create the data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

modelArch = [
    'ddim','ddim','ddim',
    'ddpm','ddpm','ddpm',
    'pndm','pndm','pndm',
    'BigGAN', 'BigGAN', 'BigGAN',
    'ContraGAN', 'ContraGAN', 'ContraGAN',
    'SNGAN', 'SNGAN', 'SNGAN'
]

# Check if CUDA is available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Modify the classifier head for our specific number of classes
num_classes = len(set(modelArch))
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model, loss function, and optimizer defined.")

# Training the model
print("Training the model...")
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    total_samples = 0 
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() * inputs.size(0)
        running_loss += batch_loss
        total_samples += inputs.size(0)  # Increment the total samples processed

        # Update the progress bar with average loss
        average_loss = running_loss / total_samples
        pbar.set_postfix(loss=average_loss)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")


