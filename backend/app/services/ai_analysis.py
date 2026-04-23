from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def _onnx_model_path() -> Path:
    return Path("models") / "deepfake_detector.onnx"


def _onnx_model_exists() -> bool:
    return _onnx_model_path().is_file()


def _prepare_onnx_input(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    # NHWC -> NCHW
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def _onnx_probability(image: np.ndarray) -> float:
    model_path = _onnx_model_path()
    net = cv2.dnn.readNetFromONNX(str(model_path))
    blob = _prepare_onnx_input(image)
    net.setInput(blob)
    output = net.forward()
    value = float(np.squeeze(output))
    probability = 1.0 / (1.0 + np.exp(-value))
    return round(float(max(0.0, min(1.0, probability))), 4)


def _cnn_proxy_probability(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_density = float(np.mean(cv2.Canny(gray, 100, 200) > 0))
    blur_score = 1.0 - _normalize(laplacian_var, 10.0, 3500.0)
    edge_score = 1.0 - _normalize(edge_density, 0.03, 0.35)
    return round(float(0.6 * blur_score + 0.4 * edge_score), 4)


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
    image = _decode_image(image_bytes, filename)

    if _onnx_model_exists():
        onnx_prob = _onnx_probability(image)
        freq_prob = _frequency_proxy_probability(image)
        results = [
            AiAnalysisResult(
                model="cnn_onnx_v1",
                fake_probability=onnx_prob,
                decision=_decision(onnx_prob),
            ),
            AiAnalysisResult(
                model="frequency_proxy_v1",
                fake_probability=freq_prob,
                decision=_decision(freq_prob),
            ),
        ]
        ensemble_probability = round((onnx_prob + freq_prob) / 2.0, 4)
        return AiAnalysisBundle(
            model_results=results,
            ensemble_probability=ensemble_probability,
            ensemble_decision=_decision(ensemble_probability),
            inference_mode="onnx+proxy",
        )

    cnn_prob = _cnn_proxy_probability(image)
    freq_prob = _frequency_proxy_probability(image)
    results = [
        AiAnalysisResult(
            model="cnn_proxy_v1",
            fake_probability=cnn_prob,
            decision=_decision(cnn_prob),
        ),
        AiAnalysisResult(
            model="frequency_proxy_v1",
            fake_probability=freq_prob,
            decision=_decision(freq_prob),
        ),
    ]
    ensemble_probability = round((cnn_prob + freq_prob) / 2.0, 4)
    return AiAnalysisBundle(
        model_results=results,
        ensemble_probability=ensemble_probability,
        ensemble_decision=_decision(ensemble_probability),
        inference_mode="proxy-only",
    )


def ai_model_status() -> dict:
    model_path = _onnx_model_path()
    return {
        "onnx_model_found": _onnx_model_exists(),
        "expected_model_path": str(model_path),
        "active_mode": "onnx+proxy" if _onnx_model_exists() else "proxy-only",
    }
