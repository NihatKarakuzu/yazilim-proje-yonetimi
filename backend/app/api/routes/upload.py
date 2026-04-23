from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.storage import save_upload_file


router = APIRouter(tags=["Upload"])


@router.post("/upload")
def upload_image(file: UploadFile = File(...)) -> dict[str, str | int]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adi bos olamaz.")

    try:
        saved_path = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_size = Path(saved_path).stat().st_size
    return {
        "message": "Dosya basariyla yuklendi.",
        "filename": saved_path.name,
        "path": str(saved_path),
        "size_bytes": file_size,
        "content_type": file.content_type or "unknown",
    }
