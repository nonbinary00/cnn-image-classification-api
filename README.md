# cnn-image-classification-api

This is my early deep learning project where I trained a simple Convolutional
Neural Network (CNN) on the FashionMNIST dataset and deployed it as a FastAPI
inference service.

The goal of this project was to:

- Understand how CNNs work
- Train a basic image classification model
- Save and load trained PyTorch models
- Build a simple inference API
- Connect ML training with backend deployment

## Model

The model is a simple Convolutional Neural Network built with PyTorch.

Architecture:

- Conv2D
- ReLU
- MaxPooling
- Conv2D
- ReLU
- MaxPooling
- Fully Connected layers

The model was trained on the FashionMNIST dataset (12 clothing classes). Based
on the training curves, the model performs best around epochs 9–11.  
Additional epochs did not significantly improve validation accuracy.

The notebook includes:

- Dataset loading
- Data normalization
- Training loop
- Evaluation
- Loss & accuracy plots

Model weights are saved to: artifacts/model.pt Class names are saved to:
artifacts/classes.json

## Running the API

The API is built with FastAPI.

### Install dependencies

```bash
pip install -r requirements.txt

Run server  uvicorn app.main:app --reload

Go to: http://127.0.0.1:8000/docs

Prediction Endpoint
POST /predict

Upload an image (JPG / PNG / WEBP).

Example response:

{
  "class": "Pullover",
  "confidence": 0.9975
}

🛠 Tech Stack

Python
PyTorch
Torchvision
FastAPI
Uvicorn
Matplotlib

What I Learned:

How convolutional layers extract features
How training and evaluation loops work in PyTorch
How to save and load models for inference
How to build a simple ML inference API
Basic debugging of backend + ML integration
```
