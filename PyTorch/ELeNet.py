#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program implements a CNN to classify images using PyTorch.

The goal of this program is to become more familiar with CNNs and
to gain familiarity with creating them in PyTorch.

The architecture of the CNN is a modified version of LeNet.
        
With this modified architecture the model obtains an accuracy of 75% on the 
famous CIFAR10 image dataset.

Example output:
    
    Epoch 94, Loss 0.045
    Epoch 95, Loss 0.042
    Epoch 96, Loss 0.042
     Epoch 97, Loss 0.044
    Epoch 98, Loss 0.039
    Epoch 99, Loss 0.041
    Epoch 100, Loss 0.040
    Training Ended
    Accuracy on test set: 75 %
    
The LeNet CNN has the following architecture:
    
    Input layer - grayscale image (1 input channel) with 6 output
    channels and a 5x5 kernel
    
    Convolution with 6 input channels, 16 output channels, and a 
    5x5 kernel
    
    Pool with 2x2 average kernel + 2 stride
    
    Convolution with 5x5 kernel
    
    Pool with 2x2 average kernel + 2 stride
    
    Dense: 120 fully connected neurons
    Dense: 84 fully connected neurons
    Dense: 10 fully connected neurons
    
More information about the LeNet architecture can be found below:
    
https://en.wikipedia.org/wiki/LeNet

For our enhanced LeNet model we have the following architecture:
    
----------------------------------------------------------------------------
          Layer (type)          |         Output Shape         | Parameters
============================================================================
          Conv2d (conv1)        |   [batch_size, 32, H, W]     |  2432  
      BatchNorm2d (bn1)         |   [batch_size, 32, H, W]     |  64   
          Conv2d (conv2)        |   [batch_size, 64, H, W]     |  51264 
      BatchNorm2d (bn2)         |   [batch_size, 64, H, W]     |  128  
       MaxPool2d (max_pool1)    |   [batch_size, 64, H/2, W/2] |  0    
       MaxPool2d (max_pool2)    |   [batch_size, 64, H/4, W/4] |  0    
           Flatten (flatten)     |  [batch_size, 1024]         |  0    
          Linear (fc1)          |   [batch_size, 512]          |  524800
          Dropout (dropout)      |  [batch_size, 512]          |  0    
          Linear (fc2)          |   [batch_size, 256]          |  131328
          Linear (fc3)          |   [batch_size, 10]           |  2570  
============================================================================
Total params: 710,586
Trainable params: 710,586
Non-trainable params: 0
----------------------------------------------------------------------------
 
Note that the code we are using as a guide on this can be found below:
    
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Architecture parameter data and table created by using ChatGPT***.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork(nn.Module):
    """
    A convolutional neural network for image classification.
    """
    def __init__(self):
        """
        Initialize the network architecture.
        
        This constructor defines the layers of the network.
        
        (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (fc1): Linear(in_features=1600, out_features=512, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
        (fc2): Linear(in_features=512, out_features=256, bias=True)
        (fc3): Linear(in_features=256, out_features=10, bias=True)
        
        """
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        
        
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        This method defines the forward propagation logic of the network.
        
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def imshow(img: torch.Tensor):
    """
    Helper function used to display an image from the training set.
    
    Args:
        img (torch.Tensor): The image tensor that will be shown, it should be normalized
    
    Returns:
        None
    """
    # Unnormalize the image (assuming it was previously normalized)
    img = img / 2 + 0.5 
    
    # Convert the tensor to a NumPy array
    npimg = img.numpy()
    
    # Transpose the NumPy array to match the dimensions that are expected by
    # matplotlib (from channels-first to channels-last)
    # Note that the tuple (1, 2, 0) indicates the order of how we want to re arange 
    # the dimensions
    npimg = np.transpose(npimg, (1, 2, 0))
    
    # Display image using matplotlib
    plt.imshow(npimg)
    plt.show()
    

def train(training_loader, epochs, neural_network):
    """
    Function that implements the training for the neural network.
    
    We first set the loss function (criterion) to be the cross
    entropy loss function.  We use stochastic gradient 
    descent as the optimizer.
    
    The number of training epochs is provided as a parameter to the 
    train function.
    
    For every 1000 batches the function will display the loss.
    
    Args:

    training_loader : PyTorch DataLoader created for the training set
    epochs : Number of epochs for the model to train through
    neural_network : CNN model defined above

    Returns:
        None

    """
    # Define loss to be the Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer to be stochastic gradient descent
    optimizer = optim.SGD(neural_network.parameters(), lr=0.001, momentum=0.9)

    # Train the neural network
    for epoch in range(epochs):
        # Variable to keep track of the total loss
        total_loss = 0.0
        # Enumerate will let us loop over the data in training_loader
        # while also keeping track of the index
        for i, data in enumerate(training_loader, 0):
            # Create variables for the inputs and labels in the data
            inputs, labels = data
            # Zero out the gradients of all parameters of the model
            optimizer.zero_grad()
            # Collect outputs determined by neural network
            outputs = neural_network(inputs)
            # Compute the error/loss
            loss = criterion(outputs, labels)
            # Back propagate the loss through the network
            loss.backward()
            # Update parameters of the model using the gradient
            # that was computed during the backward pass above
            optimizer.step()
            # Update the total loss computed
            total_loss += loss.item()
            if i % 1000 == 999: # Print out every 1000 mini-batches
                print(f'Epoch {epoch + 1}, Loss {total_loss / 1000:.3f}')
                total_loss = 0.0  
    
    print("Training Ended")
    
def test(testing_loader, neural_network):
    """
    Function that implements the testing/accuracy for the neural network.
    
    The function determines the accuracy of the CNN by computing the
    number of correctly classified images and dividing that by the
    total number of images in the testing set.
    
    Args:

    testing_loader : PyTorch DataLoader created for the testing set
    neural_network : CNN model defined above

    Returns:
        None

    """
    # Define variables for holding the number of correctly classified
    # images and the total number of images
    correct = 0
    total = 0
    # Disable the gradient during eval
    with torch.no_grad(): 
        for data in testing_loader:
            # Create variables for the images and their labels
            images, labels = data
            # Compute output
            outputs = neural_network(images)
            # Determine number of correctly classified images
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Display accuracy of model
    print('Accuracy on test set: %d %%' % (100 * correct / total))

def main():
    # Use transforms.Compose to define a series of image transformations
    transform = transforms.Compose([
    # Convert the image to a PyTorch tensor
    transforms.ToTensor(),
        # Normalize the image with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5)
        # This normalization brings the pixel values to the range [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # CIFAR10 training dataset is downloaded to '/.data', marked for training, and transformed using the transform defined above
    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # DataLoader created for the training set data with a batch size of 32 and shuffling enabled
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

    # CIFAR10 testing dataset is downloaded to './data', marked for testing, and transformed using the transform defined above
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # DataLoader created for the test set data with a batch size of 32 and shuffling disabled
    testing_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Create a list to hold the classes - used for showing sample images below
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']
    
    # Get random image
    images, labels = next(iter(training_loader))

    # Display an image
    imshow(torchvision.utils.make_grid(images[0]))
    print('Label: ', classes[labels[0].item()])
    
    # Initialize neural network with architecture defined above
    neural_network = NeuralNetwork()
    
    # Set number of epochs for training
    epochs = 10
    
    # Train CNN
    train(training_loader, epochs, neural_network)
    # Test CNN
    test(testing_loader, neural_network)
    
    # Save model
    torch.save(neural_network.state_dict(), 'model.pth')
    
main()
        