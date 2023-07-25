import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset
from torch.optim import Adam
from torch import nn, save, load
import zipfile
from torch.cuda.amp import autocast, GradScaler
import random

class RelabeledDataset(Dataset):
    def __init__(self, original_dataset, new_label):
        self.data = original_dataset
        self.new_label = new_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, _ = self.data[index]  # Assuming the dataset returns (sample, label)

        return x, self.new_label

def extract_dataset(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

def create_dataset(benign_dataset_path, malignant_dataset_path, num_benign, num_malignant):
    total_samples = num_benign + num_malignant
    half_total = total_samples // 2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize all images to 224px x 224px
        transforms.ToTensor()  # convert each image to a tensor
    ])
    benign_dataset = datasets.ImageFolder(root=benign_dataset_path, transform=transform)
    malignant_dataset = datasets.ImageFolder(root=malignant_dataset_path, transform=transform)

    benign_dataset = RelabeledDataset(benign_dataset, 0)
    malignant_dataset = RelabeledDataset(malignant_dataset, 1)

    # Generate a random array of indices for undersampling the malignant class
    malignant_indices = random.sample(range(0, 5428), num_malignant)
    malignant_dataset = torch.utils.data.Subset(malignant_dataset, malignant_indices)

    # Combine the datasets
    dataset = torch.utils.data.ConcatDataset([benign_dataset, malignant_dataset])

    # Reshuffe and recombine
    first_half, second_half = random_split(dataset, [half_total, half_total])
    dataset = torch.utils.data.ConcatDataset([first_half, second_half])
    return dataset

def split_dataset(dataset, train_ratio, test_ratio):
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])
    return train_dataset, test_dataset

def split_dataset_with_validation(dataset, train_ratio, validate_ratio, test_ratio):
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_validate_samples = int(validate_ratio * num_samples)
    num_test_samples = int(test_ratio * num_samples)

    # Split the dataset
    train_dataset, validate_dataset, test_dataset = random_split(dataset, [num_train_samples, num_validate_samples, num_test_samples])
    return train_dataset, validate_dataset, test_dataset

def set_optimiser(selected_optimiser, neuralnet, learning_rate, weight_decay):
    if (selected_optimiser == "Adam"):
        return Adam(neuralnet.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train(neuralnet, device, train_dataloader, num_epochs, loss_function, optimiser, validate_dataset=None, validate_dataloader=None):
    # Variables for storing the highest scores/best loss during validation
    highest_accuracy = 0
    highest_f1_score = 0
    lowest_loss = 1.00000

    # Count the number of epochs
    epoch_counter = 0

    for epoch in range(num_epochs): # For each Epoch
        for batch in train_dataloader:
            x,y = batch
            x, y = x.to(device), y.to(device)
            predicted_val = neuralnet(x)
            loss = loss_function(predicted_val, y)

            # Use backpropagation
            loss.backward() 
            optimiser.step()
            optimiser.zero_grad()

        print("Epoch", epoch_counter, ": Loss =", loss.item())
        
        # If there is validation
        if (validate_dataset != None):
            accuracy, f1_score = test(neuralnet, device, validate_dataset, validate_dataloader, classification_threshold, True)
            if (f1_score > highest_f1_score):
                highest_f1_score = f1_score
                with open('trained_weights.pt', 'wb') as f:      # Save the model weights
                        save(neuralnet.state_dict(), f)
        else:
            if (loss < lowest_loss):
                lowest_loss = loss
                print("The lowest loss is now", lowest_loss.item())
                with open('trained_weights.pt', 'wb') as f:      # Save the model weights
                        save(neuralnet.state_dict(), f)
        epoch_counter += 1

# Test the model
def test(neuralnet, device, test_dataset, test_dataloader, classification_threshold, is_validating):
    # Only load the model for testing after the training process
    if (is_validating == False):
        with open('trained_weights.pt', 'rb') as f:
            state_dict = torch.load(f, map_location=device)
            neuralnet.load_state_dict(state_dict)

    # Store the total number of entries and correctly-predicted output
    num_entries = len(test_dataset)

    # Store TP, TN, FP, FN
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Print out the first batch only
    first_batch_printed = False

    for batch in test_dataloader:
        # Extract values from the batch
        x,actual_values = batch 

        # Run the input into the neural network
        predicted_val = neuralnet(x.to(device))
        predicted_digits = [] # Store an array of the predicted digits for this batch
        
        # For output
        for result in predicted_val:
            class_probabilities = torch.softmax(result, dim=0)

            predicted_dig = (class_probabilities[1] >= classification_threshold).int()
            predicted_digits.append(predicted_dig)

        for i in range(len(actual_values)):
            predicted_output = predicted_digits[i]
            actual_output = actual_values[i].item()
            
            # Update the number of TPs, TNs, FPs, and FNs
            if (predicted_output == 1 and actual_output == 1):
                true_positives += 1
            elif (predicted_output == 0 and actual_output == 0):
                true_negatives += 1
            elif (predicted_output == 1 and actual_output == 0):
                false_positives += 1
            elif (predicted_output == 0 and actual_output == 1):
                false_negatives += 1
        
        # Print if its the first batch
        if (first_batch_printed is False):
            print_batch(predicted_digits, actual_values)
            first_batch_printed = True

    # Evaluate model performance
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    f1_score = (true_positives / (true_positives + 0.5 * (false_positives + false_negatives)))

    # Print the results of the test
    print("Accuracy :", accuracy)
    print("F1 Score :", f1_score)
    return accuracy, f1_score

def print_batch(predicted_digits, actual_values):
    print("-----------------FIRST BATCH-----------------")
    for i in range(len(predicted_digits)):
        predicted_output = predicted_digits[i].item()
        actual_output = actual_values[i].item()
        print("Predicted Actual: ", predicted_output, actual_output)
    print("---------------------------------------------")