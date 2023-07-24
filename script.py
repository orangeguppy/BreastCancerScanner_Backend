import torch

import helper_functions
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim import Adam
from torch import nn, save, load
import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.0001
weight_decay = 0
dropout_rate = 0
selected_optimiser = "Adam"
classification_threshold = 0.5

# param_grid = {
#     'batch_size': [16, 32, 64, 128]
#     'learning_rate': [0.0001, 0.01]
#     'weight_decay': [0]
#     'dropout_rate': [0]
#     'selected_optimiser': ["Adam"]
# }

# Dataset split ratios
train_ratio = 0.8
validate_ratio = 0
test_ratio = 0.2

model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights="DenseNet201_Weights.IMAGENET1K_V1")

# Modify the final fully connected layer for binary classification (2 classes)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 2)

model.to(device) # Move it to the GPU

# Extract the images
helper_functions.extract_dataset("breakhis-10.zip", "histology_breast")

# Build a PyTorch dataset from the extracted images
dataset = helper_functions.create_dataset("histology_breast/benign", "histology_breast/malignant", 2480, 3720)
print("The dataset has", len(dataset), "samples")

# Split the dataset into training, validating, and testing sets
if (validate_ratio == 0):
    train_dataset, test_dataset = helper_functions.split_dataset(dataset, train_ratio, test_ratio)
else:
    train_dataset, validate_dataset, test_dataset = helper_functions.split_dataset_with_validation(dataset, train_ratio, validate_ratio, test_ratio)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

# Create dataloaders for the training dataset and testing dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create the model, and then define the optimiser and loss functions
optimiser = helper_functions.set_optimiser(selected_optimiser, model, learning_rate)
loss_function = nn.CrossEntropyLoss()

# Train the model
helper_functions.train(model, device, train_dataloader, num_epochs, loss_function, optimiser)

# Test the model
helper_functions.test(model, device, test_dataset, test_dataloader, classification_threshold)