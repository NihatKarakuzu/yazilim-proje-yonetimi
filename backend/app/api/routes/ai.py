from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ai_analysis import analyze_ai_models


router = APIRouter(tags=["AI Analysis"])


@router.post("/analyze/ai")
def analyze_ai(image: UploadFile = File(...)) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Dosya içeriği boş olamaz.")

    try:
        model_results, ensemble_probability, ensemble_decision = analyze_ai_models(
            image_bytes, image.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": image.filename,
        "ensemble": {
            "fake_probability": ensemble_probability,
            "decision": ensemble_decision,
        },
        "models": [
            {
                "model": item.model,
                "fake_probability": item.fake_probability,
                "decision": item.decision,
            }
            for item in model_results
        ],
    }
