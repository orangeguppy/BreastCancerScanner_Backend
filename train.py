import neuralnetwork
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim import SGD
from torch import nn, save, load

# Creating the dataset from image files
dataset_path = "histology_breast" # define the path to the dataset
transform = transforms.Compose([ # transform the images
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Load the dataset with DataLoader
# batch_size = 128
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for images, labels in data_loader:
#     print(images, labels)

# Define the split ratios (e.g., 80% for training and 20% for testing)
train_ratio = 0.8
test_ratio = 0.2

# Calculate the number of samples for each split
num_samples = len(dataset)
num_train_samples = int(train_ratio * num_samples)
num_test_samples = num_samples - num_train_samples

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

# Create data loaders for training and testing
batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for images, labels in train_dataloader:
#     print(images, labels)
print("Done")

# Create the model, and then define the optimiser and loss functions
neuralnet = neuralnetwork.TumourClassifier().to('cpu') # running this on a CPU. Use 'cuda' for GPU
optimiser = SGD(neuralnet.parameters(), lr=0.01) # Use Stochastic Gradient Descent, and a suggested default learning rate
loss_function = nn.CrossEntropyLoss()

print("Training the model now")

num_batches = 0

# TRAIN THE MODEL:D
for epoch in range(11): # Train for 11 epochs
    print("Started epoch")
    for batch in train_dataloader:
        print("Batch ", num_batches, " started")
        x,y = batch 
        x, y = x.to('cpu'), y.to('cpu')
        print("Moved to cpu")
        predicted_val = neuralnet(x)
        loss = loss_function(predicted_val, y)

        print("Batch halfway")
        # Use backpropagation
        optimiser.zero_grad()
        loss.backward() 
        optimiser.step()
        num_batches = num_batches + 1
        print("Batch ended")
    print("Epoch done", loss.item())

# Save the model weights
with open('trained_weights.pt', 'wb') as f: 
        save(neuralnet.state_dict(), f)