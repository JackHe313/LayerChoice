import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import defaultdict, Counter
import timm
import os
from tqdm import tqdm
import time

# Define transformations for the training and testing datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Model, loss function, and optimizer defined.")
base_dir = '/home/jackhe/LayerChoice/fingerprinting_dataset_full'
num_epochs = 50
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
model = models.resnet50(pretrained=True)

# Modify the classifier head for our specific number of classes
num_classes = len(set(modelArch))
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model, loss function, and optimizer defined.")

# Training the model
print("Training the model...")
model.train()
time_start = time.time()

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0  # Track correct predictions for accuracy calculation

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()  # Increment correct predictions

        batch_loss = loss.item() * inputs.size(0)
        running_loss += batch_loss
        total_samples += inputs.size(0)  # Increment the total samples processed

        # Update the progress bar with average loss and accuracy
        average_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples * 100
        pbar.set_postfix(loss=average_loss, accuracy=f"{accuracy:.2f}%")
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions / len(train_loader.dataset) * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Training complete. Time taken: ", time.time()-time_start)

# Evaluate the model with majority vote
model.eval()
label_predictions = defaultdict(list)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        

        for label, prediction in zip(labels, predicted):
            label_predictions[label.item()].append(prediction.item())

# Perform a majority vote for each label
majority_vote_correct = 0
for label, predictions in label_predictions.items():
    if predictions:
        # Perform majority vote
        most_common_prediction, count = Counter(predictions).most_common(1)[0]
        # Check if the most common prediction is correct
        if most_common_prediction == label:
            majority_vote_correct += 1

# Calculate accuracy based on majority votes
majority_vote_accuracy = majority_vote_correct / len(label_predictions)
print(f"Majority Vote Accuracy: {majority_vote_accuracy * 100:.2f}%")





