from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.analysis_store import save_analysis_result
from app.services.classical_analysis import (
    FeatureAnalysisResult,
    analyze_with_akaze,
    analyze_with_orb,
    analyze_with_sift,
)


router = APIRouter(tags=["Classical Analysis"])


def _serialize_result(algorithm: str, result: FeatureAnalysisResult) -> dict:
    return {
        "algorithm": algorithm,
        "decision": result.decision,
        "similarity_score": result.similarity_score,
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

    response = {
        "reference_image": filename_a,
        "test_image": filename_b,
        "results": [
            _serialize_result("ORB", orb_result),
            _serialize_result("AKAZE", akaze_result),
            _serialize_result("SIFT", sift_result),
        ],
    }
    response["stored_in_db"] = save_analysis_result(
        analysis_type="classical_multi",
        payload=response,
        input_filename=filename_b,
        reference_filename=filename_a,
        decision=min(
            (
                response["results"][0]["decision"],
                response["results"][1]["decision"],
                response["results"][2]["decision"],
            ),
            key=lambda x: 0 if x == "suspicious" else 1,
        ),
        score=round(
            (
                response["results"][0]["similarity_score"]
                + response["results"][1]["similarity_score"]
                + response["results"][2]["similarity_score"]
            )
            / 3.0,
            4,
        ),
    )
    return response
