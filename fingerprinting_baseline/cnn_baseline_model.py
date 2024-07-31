import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict, Counter
import os
from tqdm import tqdm

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define transformations for the training and testing datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Model, loss function, and optimizer defined.")
base_dir = '/home/jackhe/LayerChoice/fingerprinting_dataset_small'
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
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model, loss function, and optimizer
num_classes = len(set(modelArch))
model = CNNModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
print("Training the model...")
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


