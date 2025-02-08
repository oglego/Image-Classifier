#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

def image_request(image_path: str, url: str) -> dict:
    """Sends an image file to a classification API and returns the response.

    Args:
        image_path (str): The file path of the image to be classified.
        url (str): The endpoint of the classification API.

    Returns:
        dict: The JSON response from the API, expected to contain classification results.
    """
    with open(image_path, 'rb') as img:
        files = {'file': img}  
        response = requests.post(url, files=files)  
    return response.json()  

# Define API endpoint and image path
url = 'http://localhost:8000/classify'  
image_path = 'images/cat1.png'  

# Send the image for classification and print the response
print(image_request(image_path, url))  
