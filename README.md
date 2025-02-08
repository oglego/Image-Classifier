# Image Classification Web App

This is a lightweight web application that classifies images using a deep learning model (ELeNet). The application consists of two main components:

1. **Spring Boot Backend (Java)** – Sends an image to a Python web API.
2. **Python Web API** – Uses a saved PyTorch (`model.pth`) deep learning model to classify the image and return the result.

This is the same model used in 

https://github.com/oglego/ELeNet

## Features

- Accepts image uploads (CIFAR10) for classification.
- Sends images from the Java backend to the Python API.
- Uses a pre-trained deep learning model for classification.
- Returns classification results in real-time.

## Technologies Used

- **Spring Boot (Java)** – Handles the backend logic and API communication.
- **Python (FastAPI)** – Serves the deep learning model via a web API.
- **PyTorch** – Loads and runs the model for classification.
- **REST API** – Enables communication between Java and Python services.

## Getting Started

### Prerequisites

- **Java 17+** (for the Spring Boot backend)
- **Python 3.10+** (for the API)
- **Maven** (for building the Java project)
- **PyTorch** (for running the model)
- **FastAPI** (for the Python web server)

### Installation

#### 1. Clone the Repository
```sh
git clone https://github.com/oglego/Image-Classifier.git
cd ImageClassifier
```

#### 2. Launch uvicorn
Navigate to the PyTorch directory and input the following into your terminal:
```sh
uvicorn ImageClassifier:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

#### 3. Launch Spring Boot app
Use Maven to do a clean install
```sh
mvn clean install
```

Then run
```sh
mvn spring-boot:run
```

#### 3. Launch requests
Use the ImageRequests.py file to send images to the classifier

# Image-Classifier
