from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import torch

from app.inference import load_model, load_classes, predict_image


app = FastAPI(title="CNN FashionMNIST Inference API")


device = "cuda" if torch.cuda.is_available() else "cpu"


classes = load_classes()
model = load_model(device)


# health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device
    }


# predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # проверка типа файла
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG/PNG/WEBP image."
        )

    
    try:
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to process image."
        )

    
    return predict_image(
        model=model,
        device=device,
        img=img,
        classes=classes
    )