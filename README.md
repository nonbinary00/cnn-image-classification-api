# CNN Image Classification API (End-to-End ML Project)

This project is part of my learning journey into Machine Learning Engineering.  
The goal was not only to train a CNN model, but to understand how a model can be
prepared and served in a production-like environment.

I wanted to move beyond notebooks and experiment with deploying a trained model
as an API.

## Project Goal

- Train a convolutional neural network on FashionMNIST
- Understand training and evaluation workflows in PyTorch
- Save and reload trained model weights
- Build a simple inference API using FastAPI
- Connect ML training with backend serving

This project focuses on understanding the full ML pipeline:  
**data → training → evaluation → model saving → inference API**

## Model Overview

The model is a simple CNN implemented in PyTorch.

### Architecture

- Conv2D
- ReLU
- MaxPooling
- Conv2D
- ReLU
- MaxPooling
- Fully Connected layers

The model was trained on the FashionMNIST dataset (10 classes of clothing).

Final test accuracy: ~89–91%.

From the training curves, the model achieves its best generalization performance
around epochs 9–11.  
Training beyond that does not significantly improve validation accuracy and may
start to show mild overfitting.

## Training Notebook

Training was performed in: notebooks/01_training.ipynb

The notebook includes:

- Data loading and normalization
- Training and evaluation loops
- Loss and accuracy tracking
- Visualization of learning curves
- Model checkpoint saving

## Inference API

The trained model is served through a FastAPI application.

This allowed me to understand how ML models can be:

- Loaded into memory
- Used for real-time inference
- Exposed via HTTP endpoints

### Available Endpoints

- `GET /health` – service status check
- `POST /predict` – image classification endpoint

Example response:

```json
{
  "class": "Pullover",
  "confidence": 0.9975
}
```

Tech Stack

Python PyTorch Torchvision FastAPI Uvicorn Matplotlib

Project Structure:

app/ # FastAPI inference service src/ # Model architecture notebooks/ # Training
notebook artifacts/ # Saved model weights (not tracked in Git) assets/ #
Training curves

What I Learned:

Through this project I learned: How convolutional layers extract local spatial
features How training and evaluation loops are structured in PyTorch How to
monitor model performance and detect overfitting How to save and reload model
weights for inference How to build and test a simple ML inference API Basic
debugging of ML + backend integration How to expose a trained model via HTTP
endpoints How to handle file uploads and dependency-related runtime issues in an
API service
