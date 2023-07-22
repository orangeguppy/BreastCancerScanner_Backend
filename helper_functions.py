# import neuralnetwork
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim import Adam
from torch import nn, save, load
import zipfile
from torch.cuda.amp import autocast, GradScaler
import random

def create_dataset(benign_dataset_path, malignant_dataset_path, num_benign, num_malignant):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize all images to 224px x 224px
        transforms.ToTensor()  # convert each image to a tensor
    ])
    benign_dataset = datasets.ImageFolder(root=benign_dataset_path, transform=transform)
    malignant_dataset = datasets.ImageFolder(root=malignant_dataset_path, transform=transform)

    # Generate a random array of indices for undersampling the malignant class
    malignant_indices = random.sample(range(0, 5428), num_malignant)
    malignant_dataset = torch.utils.data.Subset(malignant_dataset, malignant_indices)

    # Combine the datasets
    dataset = torch.utils.data.ConcatDataset([benign_dataset, malignant_dataset])
    return dataset

def split_dataset(dataset, train_ratio, test_ratio):
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])
    return train_dataset, test_dataset

def set_optimiser(selected_optimiser, neuralnet, learning_rate):
    if (selected_optimiser == "Adam"):
        return Adam(neuralnet.parameters(), lr=learning_rate)

def train(neuralnet, device, train_dataloader, num_epochs, loss_function, optimiser):
    num_batches = 0
    best_rate = 1.0
    for epoch in range(num_epochs): # Train for 11 epochs
        print("Started epoch")
        for batch in train_dataloader:
            print("Batch ", num_batches, " started")
            x,y = batch 
            x, y = x.to(device), y.to(device)
            predicted_val = neuralnet(x)
            loss = loss_function(predicted_val, y)

            print("Batch halfway")
            # Use backpropagation
            loss.backward() 
            optimiser.step()
            optimiser.zero_grad()
            num_batches = num_batches + 1
            print("Batch ended")
        print("Epoch done", loss.item())
        if (loss.item() < best_rate):
            best_rate = loss.item()
            with open('trained_weights.pt', 'wb') as f:      # Save the model weights
                    save(neuralnet.state_dict(), f)

# Test the model
def test(neuralnet, device, test_dataset, test_dataloader):
    # Test the model
    with open('trained_weights.pt', 'rb') as f:
        state_dict = torch.load(f, map_location=device)
        neuralnet.load_state_dict(state_dict)

    # Store the total number of entries and correctly-predicted output
    num_entries = len(test_dataset)
    num_correct = 0

    # Print out the first batch only
    first_batch_printed = False

    for batch in test_dataloader:
        # Extract values from the batch
        x,actual_values = batch 

        # Run the input into the neural network
        # predicted_val = neuralnet(x) # Raw data from the network
        predicted_val = neuralnet(x.to(device))
        predicted_digits = [] # Store an array of the predicted digits for this batch
        
        # For output
        for result in predicted_val: # Each 'result' is an array of 10 values, for instance the first element of result
                                    # stores the probability that the image has the digit '0', index 1 for P(digit is 1), etc
            predicted_dig = torch.argmax(result).item()
            predicted_digits.append(predicted_dig)

        # Compare the input and output
        if (first_batch_printed is False): print("-----------------FIRST BATCH STARTED-----------------")
        for i in range(len(actual_values)):
            predicted_output = predicted_digits[i]
            actual_output = actual_values[i].item()

            # Check if the output is correct
            if (predicted_output == actual_output):
                num_correct += 1

            # Print out the results if it's the first batch
            if (first_batch_printed is False):
                print("Predicted Actual: ", predicted_output, actual_output)
        
        if (first_batch_printed is False): print("-----------------FIRST BATCH ENDED, ONLY THIS BATCH IS SHOWN-----------------")
        first_batch_printed = True

    # Print the results of the test
    print("NUMBER OF ENTRIES: ", num_entries)
    print("NUMBER OF CORRECT ENTRIES: ", num_correct)
    print("ACCURACY RATE: ", num_correct / num_entries)