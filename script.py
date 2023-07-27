import torch

import helper_functions
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim import Adam
from torch import nn, save, load
import zipfile
from itertools import product
from custom_densenet201 import CustomDenseNet201
import mlflow
from databricks_credentials import expt_url

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
# batch_size = 32
# num_epochs = 10
# learning_rate = 0.0001
# weight_decay = 0
# dropout_rate = 0
# selected_optimiser = "Adam"
classification_threshold = 0.5

param_grid = {
    'selected_optimiser': ["Adam", "SGD"],
    'weight_decay': [0, 0.01, 0.001],
    'dropout_rate': [0, 0.05, 0.1, 0.15, 0.2],
    'batch_size': [16, 32, 64, 128],
    'learning_rate': [0.0001, 0.01]
}

# Generate all combinations of hyperparameters
all_combinations = list(product(*param_grid.values()))

# Dataset split ratios
train_ratio = 0.8
validate_ratio = 0
test_ratio = 0.2

# Extract the images
# helper_functions.extract_dataset("breakhis-10.zip", "histology_breast")

# Build a PyTorch dataset from the extracted images
dataset = helper_functions.create_dataset("histology_breast/benign", "histology_breast/malignant", 2480, 3720)
print("The dataset has", len(dataset), "samples")

# Start an MLflow experiment and make it active
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(expt_url)

# Iterate through each combination
for combination in all_combinations:
    # Store the current combination of hyperparameters
    hyperparameters = dict(zip(param_grid.keys(), combination))

    # Generate a run name
    name_parts = [f"{key}_{value}" for key, value in hyperparameters.items()]
    run_name = "_".join(name_parts)

    # Begin a run for this combination
    mlflow.start_run(run_name=run_name)

    # Log the hyperparameters
    mlflow.log_params(hyperparameters)
    mlflow.log_param("train_ratio", train_ratio)
    mlflow.log_param("test_ratio", test_ratio)
    mlflow.log_param("validate_ratio", validate_ratio)

    # Log the classification threshold
    mlflow.log_param("classification_threshold", classification_threshold)

    # Split the dataset into training, validating, and testing sets
    if (validate_ratio == 0):
        train_dataset, test_dataset = helper_functions.split_dataset(dataset, train_ratio, test_ratio)
    else:
        train_dataset, validate_dataset, test_dataset = helper_functions.split_dataset_with_validation(dataset, train_ratio, validate_ratio, test_ratio)
        validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)

    # Create dataloaders for the training dataset and testing dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
    
    # Create the model, and then define the optimiser and loss functions
    model = CustomDenseNet201(dropout_prob=hyperparameters["dropout_rate"])
    model.to(device) # Move it to the GPU/CPU
    optimiser = helper_functions.set_optimiser(hyperparameters["selected_optimiser"], model, hyperparameters["learning_rate"], hyperparameters["weight_decay"])
    loss_function = nn.CrossEntropyLoss()

    # Train the model
    loss = helper_functions.train(model, device, train_dataloader, 11, loss_function, optimiser)
    mlflow.log_metric("loss", loss)

    # Test the model
    accuracy, f1_score = helper_functions.test(model, device, test_dataset, test_dataloader, classification_threshold, False)

    # Log model performance metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_score)

    # Load the model weights from the saved .pt file
    model.load_state_dict(torch.load("trained_weights.pt"))

    # Log the model as an artifact in MLflow
    mlflow.pytorch.log_model(model, artifact_path="densenet201_models")

    # End the run
    mlflow.end_run()