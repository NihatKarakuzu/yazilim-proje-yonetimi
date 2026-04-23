from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureAnalysisResult:
    decision: str
    similarity_score: float
    keypoints_image_a: int
    keypoints_image_b: int
    good_matches: int
    total_matches: int


def _decode_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"'{filename}' gecerli bir goruntu dosyasi degil.")
    return image


def _analyze_with_detector(
    image_a_bytes: bytes,
    image_b_bytes: bytes,
    filename_a: str,
    filename_b: str,
    detector_name: str,
) -> FeatureAnalysisResult:
    image_a = _decode_image(image_a_bytes, filename_a)
    image_b = _decode_image(image_b_bytes, filename_b)

    if detector_name == "ORB":
        detector = cv2.ORB_create(nfeatures=1200)
        distance_limit = 50
        matcher_norm = cv2.NORM_HAMMING
    elif detector_name == "AKAZE":
        detector = cv2.AKAZE_create()
        distance_limit = 45
        matcher_norm = cv2.NORM_HAMMING
    elif detector_name == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("Bu OpenCV surumunde SIFT destegi bulunamadi.")
        detector = cv2.SIFT_create(nfeatures=1200)
        distance_limit = 250
        matcher_norm = cv2.NORM_L2
    else:
        raise ValueError("Bilinmeyen algoritma secimi.")

    keypoints_a, descriptors_a = detector.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(image_b, None)

    if descriptors_a is None or descriptors_b is None:
        raise ValueError("Goruntulerde yeterli ozellik noktasi bulunamadi.")

    matcher = cv2.BFMatcher(matcher_norm, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)

    if not matches:
        raise ValueError("Goruntuler arasinda eslesme bulunamadi.")

    good_matches = [m for m in matches if m.distance < distance_limit]
    similarity_score = len(good_matches) / max(len(matches), 1)
    decision = "authentic_like" if similarity_score >= 0.25 else "suspicious"

    return FeatureAnalysisResult(
        decision=decision,
        similarity_score=round(similarity_score, 4),
        keypoints_image_a=len(keypoints_a),
        keypoints_image_b=len(keypoints_b),
        good_matches=len(good_matches),
        total_matches=len(matches),
    )


def analyze_with_orb(
    image_a_bytes: bytes, image_b_bytes: bytes, filename_a: str, filename_b: str
) -> FeatureAnalysisResult:
    return _analyze_with_detector(
        image_a_bytes, image_b_bytes, filename_a, filename_b, detector_name="ORB"
    )


def analyze_with_akaze(
    image_a_bytes: bytes, image_b_bytes: bytes, filename_a: str, filename_b: str
) -> FeatureAnalysisResult:
    return _analyze_with_detector(
        image_a_bytes, image_b_bytes, filename_a, filename_b, detector_name="AKAZE"
    )


def analyze_with_sift(
    image_a_bytes: bytes, image_b_bytes: bytes, filename_a: str, filename_b: str
) -> FeatureAnalysisResult:
    return _analyze_with_detector(
        image_a_bytes, image_b_bytes, filename_a, filename_b, detector_name="SIFT"
    )
