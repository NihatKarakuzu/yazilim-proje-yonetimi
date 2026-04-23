from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import UploadFile


@dataclass
class OrbAnalysisResult:
    decision: str
    similarity_score: float
    keypoints_image_a: int
    keypoints_image_b: int
    good_matches: int
    total_matches: int


def _read_image_from_upload(file: UploadFile) -> np.ndarray:
    raw_bytes = file.file.read()
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"'{file.filename}' gecerli bir goruntu dosyasi degil.")
    return image


def analyze_with_orb(file_a: UploadFile, file_b: UploadFile) -> OrbAnalysisResult:
    image_a = _read_image_from_upload(file_a)
    image_b = _read_image_from_upload(file_b)

    orb = cv2.ORB_create(nfeatures=1200)
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)

    if descriptors_a is None or descriptors_b is None:
        raise ValueError("Goruntulerde yeterli ozellik noktasi bulunamadi.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)

    if not matches:
        raise ValueError("Goruntuler arasinda eslesme bulunamadi.")

    good_matches = [m for m in matches if m.distance < 50]
    similarity_score = len(good_matches) / max(len(matches), 1)
    decision = "authentic_like" if similarity_score >= 0.25 else "suspicious"

    return OrbAnalysisResult(
        decision=decision,
        similarity_score=round(similarity_score, 4),
        keypoints_image_a=len(keypoints_a),
        keypoints_image_b=len(keypoints_b),
        good_matches=len(good_matches),
        total_matches=len(matches),
    )
