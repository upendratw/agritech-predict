# agritech-predict/backend/advice.py

from fastapi import APIRouter, Query
from typing import Optional
import mysql.connector

from db_conn import get_db_connection

router = APIRouter()

@router.get("/treatment-advice")
def get_treatment_advice(
    crop: str = Query(..., description="Crop name"),
    label: str = Query(..., description="Detected disease label"),
    season: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
):
    """
    Fetch treatment advice for a detected crop disease
    """

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT advice
        FROM crop_disease_advice
        WHERE crop = %s
          AND label = %s
          AND (%s IS NULL OR season = %s)
          AND (%s IS NULL OR year = %s)
        ORDER BY year DESC
        LIMIT 1
    """

    cursor.execute(
        query,
        (crop, label, season, season, year, year)
    )

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        return {
            "crop": crop,
            "label": label,
            "advice": "No specific treatment advice available. Consult local agriculture officer."
        }

    return {
        "crop": crop,
        "label": label,
        "advice": row["advice"]
    }