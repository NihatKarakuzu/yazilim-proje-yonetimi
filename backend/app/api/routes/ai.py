from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.ai_analysis import ai_model_status, analyze_ai_models


router = APIRouter(tags=["AI Analysis"])


def _ai_explanation(decision: str, probability: float, results: list = None) -> str:
    suspicious_count = sum(1 for r in results if r.decision == "suspicious") if results else 0
    total_models = len(results) if results else 3
    
    if decision == "suspicious":
        return (
            f"Analiz edilen {total_models} yöntemden {suspicious_count} tanesi şüpheli bulgular tespit etti. "
            f"Genel sahtelik riski %{probability*100:.0f} olarak belirlendi."
        )
    
    trust_count = total_models - suspicious_count
    return (
        f"Analiz edilen {total_models} yöntemin {trust_count} tanesi görselin orijinal olduğunu doğrulamaktadır. "
        f"Görsel %{(1.0 - probability)*100:.0f} oranında güvenli görünüyor."
    )


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

    # Model ID mapping for humans
    name_map = {
        "cnn_simple_v1": "Model A (Hızlı CNN)",
        "resnet18_v1": "Model B (Derin ResNet)",
        "frequency_analiz_v1": "Frekans Analizi",
    }

    response = {
        "filename": image.filename,
        "inference_mode": bundle.inference_mode,
        "ensemble": {
            "fake_probability": bundle.ensemble_probability,
            "decision": bundle.ensemble_decision,
            "explanation": _ai_explanation(
                bundle.ensemble_decision, bundle.ensemble_probability, bundle.model_results
            ),
        },
        "models": [
            {
                "model": name_map.get(item.model, item.model),
                "fake_probability": item.fake_probability,
                "decision": item.decision,
            }
            for item in bundle.model_results
        ],
    }
    return response
