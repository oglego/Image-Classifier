#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import io
import torch.nn.functional as F

# Create an instance of the FastAPI application.
# This serves as the main entry point for defining routes and handling requests.
app = FastAPI()

# Define our neural network architecture - this is taken from ELeNet.
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

# Define a transformation pipeline for preprocessing images before passing them
# to the ELeNet model API endpoint
transform = transforms.Compose([
    transforms.Resize((115, 16)), 
    transforms.ToTensor(),
    # Normalize the image - these are the recommended values for CIFAR10
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

# Load the model created from training ELeNet
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classifies an uploaded image using a trained model (ELeNet).

    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: A dictionary containing the predicted class of the image.
    """
    
    # Read the uploaded image 
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)) 

    # Transform the image
    tensor = transform(image).unsqueeze(0)  

    # Perform inference 
    with torch.no_grad():
        outputs = model(tensor)  
        _, predicted = torch.max(outputs, 1)

    # Return predicted class
    return {"class": predicted.item()}



