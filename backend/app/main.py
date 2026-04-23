from fastapi import FastAPI

from app.api.routes.classical import router as classical_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router


app = FastAPI(
    title="Goruntu Sahteciligi Tespit API",
    version="0.1.0",
    description="Donem projesi icin temel backend servisleri.",
)

app.include_router(health_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(classical_router, prefix="/api")
