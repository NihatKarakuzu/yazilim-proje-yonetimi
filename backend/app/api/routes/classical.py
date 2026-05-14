from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.analysis_store import save_analysis_result
from app.services.classical_analysis import (
    FeatureAnalysisResult,
    analyze_with_akaze,
    analyze_with_orb,
    analyze_with_sift,
    analyze_with_surf,
)


router = APIRouter(tags=["Classical Analysis"])


def _decision_message(decision: str, score: float) -> str:
    if decision == "suspicious":
        return (
            f"Görüntüler arasında %{(1.0 - score)*100:.0f} oranında anlamlı farklar tespit edildi. "
            "İçerik üzerinde oynama yapılmış olma ihtimali yüksektir."
        )
    return (
        f"Görüntüler %{score*100:.0f} oranında tutarlı görünüyor. İçerik orijinal kabul edilebilir."
    )


def _serialize_result(algorithm: str, result: FeatureAnalysisResult) -> dict:
    # Algorithm name mapping
    name_map = {
        "ORB": "Hızlı Tarama",
        "AKAZE": "Detay Analizi",
        "SIFT": "Hassas Karşılaştırma",
        "SURF": "Hızlandırılmış Analiz (SURF)",
    }
    return {
        "algorithm_key": algorithm,
        "algorithm": name_map.get(algorithm, algorithm),
        "decision": result.decision,
        "similarity_score": result.similarity_score,
        "explanation": _decision_message(result.decision, result.similarity_score),
        "metrics": {
            "keypoints_image_a": result.keypoints_image_a,
            "keypoints_image_b": result.keypoints_image_b,
            "good_matches": result.good_matches,
            "total_matches": result.total_matches,
        },
    }


@router.post("/analyze/orb")
def analyze_orb(
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...),
) -> dict:
    if not reference_image.filename or not test_image.filename:
        raise HTTPException(status_code=400, detail="İki dosya da gönderilmelidir.")

    reference_bytes = reference_image.file.read()
    test_bytes = test_image.file.read()

    try:
        result = analyze_with_orb(
            reference_bytes,
            test_bytes,
            reference_image.filename,
            test_image.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = _serialize_result("ORB", result)
    response["stored_in_db"] = save_analysis_result(
        analysis_type="orb",
        payload=response,
        input_filename=test_image.filename,
        reference_filename=reference_image.filename,
        decision=result.decision,
        score=result.similarity_score,
    )
    return response


@router.post("/analyze/classical")
def analyze_classical(
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...),
) -> dict:
    if not reference_image.filename or not test_image.filename:
        raise HTTPException(status_code=400, detail="İki dosya da gönderilmelidir.")

    reference_bytes = reference_image.file.read()
    test_bytes = test_image.file.read()

    filename_a = reference_image.filename
    filename_b = test_image.filename

    try:
        orb_result = analyze_with_orb(reference_bytes, test_bytes, filename_a, filename_b)
        akaze_result = analyze_with_akaze(
            reference_bytes, test_bytes, filename_a, filename_b
        )
        sift_result = analyze_with_sift(reference_bytes, test_bytes, filename_a, filename_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    surf_result = None
    try:
        surf_result = analyze_with_surf(reference_bytes, test_bytes, filename_a, filename_b)
    except ValueError:
        # opencv-contrib olmayan kurulumlarda SURF yok; ORB/AKAZE/SIFT ile devam
        pass

    serialized = [
        _serialize_result("ORB", orb_result),
        _serialize_result("AKAZE", akaze_result),
        _serialize_result("SIFT", sift_result),
    ]
    if surf_result is not None:
        serialized.append(_serialize_result("SURF", surf_result))

    response = {
        "reference_image": filename_a,
        "test_image": filename_b,
        "surf_available": surf_result is not None,
        "results": serialized,
    }
    suspicious_count = sum(1 for item in response["results"] if item["decision"] == "suspicious")
    response["summary"] = {
        "decision": "suspicious" if suspicious_count >= 2 else "authentic_like",
        "explanation": (
            "Algoritmaların çoğu şüpheli sonuç üretti. Görüntüde değişiklik olasılığı yüksektir."
            if suspicious_count >= 2
            else "Analiz edilen yöntemlerin çoğu görüntünün orijinal olduğunu doğrulamaktadır."
        ),
    }
    if surf_result is None:
        response["summary"]["explanation"] += (
            " Not: SURF bu OpenCV kurulumunda kullanılamadı; özet ORB, AKAZE ve SIFT ile hesaplandı."
        )
    response["stored_in_db"] = save_analysis_result(
        analysis_type="classical_multi",
        payload=response,
        input_filename=filename_b,
        reference_filename=filename_a,
        decision=response["summary"]["decision"],
        score=round(
            sum(item["similarity_score"] for item in response["results"])
            / len(response["results"]),
            4,
        ),
    )
    return response
