import json
import os
from datetime import datetime

import psycopg


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _ensure_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id BIGSERIAL PRIMARY KEY,
                analysis_type VARCHAR(40) NOT NULL,
                input_filename VARCHAR(255),
                reference_filename VARCHAR(255),
                decision VARCHAR(40),
                score DOUBLE PRECISION,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    conn.commit()


def save_analysis_result(
    *,
    analysis_type: str,
    payload: dict,
    input_filename: str | None = None,
    reference_filename: str | None = None,
    decision: str | None = None,
    score: float | None = None,
) -> bool:
    db_url = _database_url()
    if not db_url:
        return False

    try:
        with psycopg.connect(db_url) as conn:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analysis_results
                    (analysis_type, input_filename, reference_filename, decision, score, payload_json)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    (
                        analysis_type,
                        input_filename,
                        reference_filename,
                        decision,
                        score,
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
            conn.commit()
        return True
    except Exception:
        return False


def list_recent_results(limit: int = 20) -> list[dict]:
    db_url = _database_url()
    if not db_url:
        return []

    try:
        with psycopg.connect(db_url) as conn:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, analysis_type, input_filename, reference_filename,
                           decision, score, payload_json, created_at
                    FROM analysis_results
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "analysis_type": row[1],
                "input_filename": row[2],
                "reference_filename": row[3],
                "decision": row[4],
                "score": row[5],
                "payload": row[6],
                "created_at": row[7].isoformat()
                if isinstance(row[7], datetime)
                else str(row[7]),
            }
            for row in rows
        ]
    except Exception:
        return []
