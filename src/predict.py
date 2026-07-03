import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
import json

from custom_model import emotionModel

EMOTION_LABELS = {0: "Angry", 1: "Happy", 2: "Sad"}

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_model(model_dir: str = "model") -> tuple[emotionModel, dict]:
    """Load model weights and normalization stats from model_dir."""
    model_dir = Path(model_dir)

    stats_path = model_dir / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"normalization_stats.json not found at {stats_path}. "
            "Run model_training.py first."
        )
    with open(stats_path) as f:
        stats = json.load(f)

    model_path = model_dir / "best_emotion_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"best_emotion_model.pth not found at {model_path}. "
            "Run model_training.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = emotionModel(num_classes=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, stats, device


def preprocess_face(face_roi: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """
    Replicate emotionDataset.__getitem__:
    face_roi (uint8 grayscale crop from OpenCV) ->
    resize to 48x48 -> PIL 'L' image -> ToTensor -> Normalize
    """
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    pil_image = Image.fromarray(face_roi, mode="L")
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def predict(image_bytes: bytes, model: emotionModel, stats: dict, device: torch.device) -> dict:
    """
    raw image bytes -> face detection -> preprocessing -> model -> result dict
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Could not decode image. Ensure the upload is a valid JPEG/PNG.")

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return {
            "emotion": None,
            "confidence": None,
            "probabilities": None,
            "message": "No face detected in the image. Please upload a clear frontal face photo.",
        }

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray_image[y:y + h, x:x + w]

    tensor = preprocess_face(face_roi, stats["Mean"], stats["Standard Deviation"])
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_class = predicted_class.item()
    confidence = round(confidence.item() * 100, 2)

    prob_dict = {
        EMOTION_LABELS[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(3)
    }

    return {
        "emotion": EMOTION_LABELS[predicted_class],
        "confidence": f"{confidence}%",
        "probabilities": prob_dict,
        "message": "Success",
    }


# Quick local test — run `python predict.py path/to/image.jpg` to verify
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    model, stats, device = load_model(model_dir="model")
    result = predict(image_path.read_bytes(), model, stats, device)
    print(result)