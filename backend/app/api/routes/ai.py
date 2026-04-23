from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.analysis_store import save_analysis_result
from app.services.ai_analysis import ai_model_status, analyze_ai_models


router = APIRouter(tags=["AI Analysis"])


@router.get("/ai/model-status")
def model_status() -> dict:
    return ai_model_status()


@router.post("/analyze/ai")
def analyze_ai(image: UploadFile = File(...)) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Dosya içeriği boş olamaz.")

    try:
        bundle = analyze_ai_models(image_bytes, image.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = {
        "filename": image.filename,
        "inference_mode": bundle.inference_mode,
        "ensemble": {
            "fake_probability": bundle.ensemble_probability,
            "decision": bundle.ensemble_decision,
        },
        "models": [
            {
                "model": item.model,
                "fake_probability": item.fake_probability,
                "decision": item.decision,
            }
            for item in bundle.model_results
        ],
    }
    response["stored_in_db"] = save_analysis_result(
        analysis_type="ai",
        payload=response,
        input_filename=image.filename,
        decision=bundle.ensemble_decision,
        score=bundle.ensemble_probability,
    )
    return response
