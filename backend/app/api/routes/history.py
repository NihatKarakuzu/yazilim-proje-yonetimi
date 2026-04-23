from fastapi import APIRouter, Query

from app.services.analysis_store import list_recent_results


router = APIRouter(tags=["Analysis History"])


@router.get("/analysis/history")
def analysis_history(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    rows = list_recent_results(limit=limit)
    return {
        "count": len(rows),
        "items": rows,
    }
