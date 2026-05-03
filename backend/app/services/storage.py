from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def ensure_upload_dir() -> Path:
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Desteklenmeyen dosya uzantısı. İzin verilenler: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    return extension


def build_unique_filename(extension: str) -> str:
    return f"{uuid4().hex}{extension}"


def save_upload_file(file: UploadFile) -> Path:
    extension = validate_extension(file.filename or "")
    upload_dir = ensure_upload_dir()
    target = upload_dir / build_unique_filename(extension)
    content = file.file.read()
    target.write_bytes(content)
    return target
