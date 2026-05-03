from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

@dataclass
class AiAnalysisResult:
    model: str
    fake_probability: float
    decision: str


@dataclass
class AiAnalysisBundle:
    model_results: list[AiAnalysisResult]
    ensemble_probability: float
    ensemble_decision: str
    inference_mode: str


# --- Model Architectures ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_resnet_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

# --- Global Model Instances (Lazy Load) ---
MODELS = {
    "simple_cnn": None,
    "resnet18": None
}

def _load_torch_models():
    if MODELS["simple_cnn"] is not None:
        return
    
    device = torch.device("cpu") # Server-side CPU is safer/easier for simple inference
    
    # Simple CNN
    try:
        m1 = SimpleCNN()
        path1 = Path("models") / "simple_cnn.pth"
        if path1.exists():
            m1.load_state_dict(torch.load(str(path1), map_location=device))
        m1.eval()
        MODELS["simple_cnn"] = m1
    except Exception as e:
        print(f"Error loading SimpleCNN: {e}")

    # ResNet18
    try:
        m2 = get_resnet_model()
        path2 = Path("models") / "resnet18_forgery.pth"
        if path2.exists():
            m2.load_state_dict(torch.load(str(path2), map_location=device))
        m2.eval()
        MODELS["resnet18"] = m2
    except Exception as e:
        print(f"Error loading ResNet18: {e}")

def _decode_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"'{filename}' geçerli bir görüntü dosyası değil.")
    return image


def _normalize(value: float, min_value: float, max_value: float) -> float:
    value = max(min_value, min(value, max_value))
    return (value - min_value) / (max_value - min_value)


def _decision(probability: float) -> str:
    return "suspicious" if probability >= 0.5 else "authentic_like"


def _torch_inference(model, image_bytes: bytes, resize_to=(32, 32)) -> float:
    if model is None:
        return 0.5 # Default neutral if failed to load
    
    try:
        transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
        return round(float(prob), 4)
    except Exception:
        return 0.5


def _frequency_proxy_probability(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1.0)
    low_band = float(np.mean(magnitude[gray.shape[0] // 3 : 2 * gray.shape[0] // 3, gray.shape[1] // 3 : 2 * gray.shape[1] // 3]))
    full_band = float(np.mean(magnitude))
    ratio = low_band / max(full_band, 1e-6)
    probability = 1.0 - _normalize(ratio, 0.85, 1.25)
    return round(float(probability), 4)


def analyze_ai_models(image_bytes: bytes, filename: str) -> AiAnalysisBundle:
    _load_torch_models()
    image = _decode_image(image_bytes, filename)

    # 1. Simple CNN Analysis
    cnn_prob = _torch_inference(MODELS["simple_cnn"], image_bytes, resize_to=(32, 32))
    
    # 2. ResNet18 Analysis (ResNet usually uses 224x224 or 32x32 depending on training)
    # Looking at train_dual_models.py, it used 32x32 for both!
    resnet_prob = _torch_inference(MODELS["resnet18"], image_bytes, resize_to=(32, 32))
    
    # 3. Frequency Analysis
    freq_prob = _frequency_proxy_probability(image)

    results = [
        AiAnalysisResult(model="cnn_simple_v1", fake_probability=cnn_prob, decision=_decision(cnn_prob)),
        AiAnalysisResult(model="resnet18_v1", fake_probability=resnet_prob, decision=_decision(resnet_prob)),
        AiAnalysisResult(model="frequency_analiz_v1", fake_probability=freq_prob, decision=_decision(freq_prob)),
    ]

    # Weighted ensemble or consensus
    # For consensus, let's see how many say suspicious
    suspicious_count = sum(1 for r in results if r.decision == "suspicious")
    
    # Probability can be the average
    ensemble_probability = round((cnn_prob + resnet_prob + freq_prob) / 3.0, 4)
    
    # Decision by majority (2/3)
    ensemble_decision = "suspicious" if suspicious_count >= 2 else "authentic_like"

    return AiAnalysisBundle(
        model_results=results,
        ensemble_probability=ensemble_probability,
        ensemble_decision=ensemble_decision,
        inference_mode="torch-comparative",
    )


def ai_model_status() -> dict:
    _load_torch_models()
    return {
        "simple_cnn_loaded": MODELS["simple_cnn"] is not None,
        "resnet18_loaded": MODELS["resnet18"] is not None,
        "active_mode": "torch-comparative",
    }
