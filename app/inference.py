import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN

# корень проекта: .../cnn-image-classification-api
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pt"
CLASSES_PATH = ARTIFACTS_DIR / "classes.json"

# те же transforms, что в обучении
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_classes() -> list[str]:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(device: str) -> torch.nn.Module:
    model = SimpleCNN(num_classes=10)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: torch.nn.Module, device: str, img: Image.Image, classes: list[str]) -> dict:
    x = _transform(img).unsqueeze(0).to(device)  # [1, 1, 28, 28]
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]      # [10]

    conf, idx = torch.max(probs, dim=0)

    return {
        "class": classes[int(idx)],
        "confidence": float(conf.item())
    }