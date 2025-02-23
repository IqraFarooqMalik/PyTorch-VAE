import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
import os

# Define the path where you want to store the dataset
data_path = './data'

# Define your transforms (example)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Initialize the CelebA dataset
dataset = CelebA(root=data_path, split='train', transform=transform, download=True)

# Check the length of the dataset (number of images)
print(f'Total number of images in the dataset: {len(dataset)}')

# Load a single sample to test if the data is accessible
image, label = dataset[0]
print(f'Image shape: {image.shape}, Label: {label}')

# Create DataLoader to batch the data
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Check if the DataLoader works (get a single batch)
for i, (images, labels) in enumerate(train_loader):
    print(f'Batch {i + 1}:')
    print(f'Images shape: {images.shape}, Labels: {labels.shape}')
    if i == 0:  # Test one batch
        break
