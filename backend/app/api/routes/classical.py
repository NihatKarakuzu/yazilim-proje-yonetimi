from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.classical_analysis import analyze_with_orb


router = APIRouter(tags=["Classical Analysis"])


@router.post("/analyze/orb")
def analyze_orb(
    reference_image: UploadFile = File(...),
    test_image: UploadFile = File(...),
) -> dict:
    if not reference_image.filename or not test_image.filename:
        raise HTTPException(status_code=400, detail="Iki dosya da gonderilmelidir.")

    try:
        result = analyze_with_orb(reference_image, test_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "algorithm": "ORB",
        "decision": result.decision,
        "similarity_score": result.similarity_score,
        "metrics": {
            "keypoints_image_a": result.keypoints_image_a,
            "keypoints_image_b": result.keypoints_image_b,
            "good_matches": result.good_matches,
            "total_matches": result.total_matches,
        },
    }
