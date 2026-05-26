from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from settings.config import ANALYSIS_TTL_HOURS, INFO_DIR


GRADE_STORE: Dict[str, Dict[str, Any]] = {}


def dataframe_to_session_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    safe_df = df.where(pd.notnull(df), None)
    return safe_df.to_dict(orient="records")


def session_records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def safe_student_dirname(student_id: str) -> str:
    return "".join(ch for ch in str(student_id) if ch.isalnum() or ch in ("-", "_"))


def get_student_info_dir(student_id: str) -> str:
    student_dir = os.path.join(INFO_DIR, safe_student_dirname(student_id))
    os.makedirs(student_dir, exist_ok=True)
    return student_dir


def get_student_file_paths(student_id: str) -> Dict[str, str]:
    student_dir = get_student_info_dir(student_id)
    return {
        "dir": student_dir,
        "grades": os.path.join(student_dir, "grades.json"),
        "analysis": os.path.join(student_dir, "analysis.json"),
        "meta": os.path.join(student_dir, "meta.json"),
    }


def write_json_file(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json_file(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_student_grades(student_id: str, grade_df: pd.DataFrame) -> None:
    paths = get_student_file_paths(student_id)
    grades_records = dataframe_to_session_records(grade_df)

    write_json_file(
        paths["grades"],
        {
            "student_id": student_id,
            "saved_at": datetime.utcnow().isoformat(),
            "row_count": len(grades_records),
            "grades": grades_records,
        },
    )

    meta = read_json_file(paths["meta"]) or {}
    meta["student_id"] = student_id
    meta["last_grade_saved_at"] = datetime.utcnow().isoformat()
    meta["row_count"] = len(grades_records)
    write_json_file(paths["meta"], meta)


def load_student_grades(student_id: str) -> pd.DataFrame:
    paths = get_student_file_paths(student_id)
    payload = read_json_file(paths["grades"])
    if not payload:
        return pd.DataFrame()

    return session_records_to_dataframe(payload.get("grades", []))


def is_analysis_cache_valid(student_id: str) -> bool:
    paths = get_student_file_paths(student_id)
    payload = read_json_file(paths["analysis"])
    if not payload:
        return False

    created_at = payload.get("created_at")
    if not created_at:
        return False

    try:
        created_dt = datetime.fromisoformat(created_at)
    except Exception:
        return False

    expires_at = created_dt + timedelta(hours=ANALYSIS_TTL_HOURS)
    return datetime.utcnow() < expires_at


def load_cached_analysis(student_id: str) -> Optional[dict]:
    if not is_analysis_cache_valid(student_id):
        return None

    paths = get_student_file_paths(student_id)
    payload = read_json_file(paths["analysis"])
    if not payload:
        return None

    return payload.get("analysis")


def save_analysis_cache(student_id: str, analysis: dict) -> None:
    paths = get_student_file_paths(student_id)

    write_json_file(
        paths["analysis"],
        {
            "student_id": student_id,
            "created_at": datetime.utcnow().isoformat(),
            "ttl_hours": ANALYSIS_TTL_HOURS,
            "analysis": analysis,
        },
    )

    meta = read_json_file(paths["meta"]) or {}
    meta["student_id"] = student_id
    meta["last_analysis_saved_at"] = datetime.utcnow().isoformat()
    meta["analysis_ttl_hours"] = ANALYSIS_TTL_HOURS
    write_json_file(paths["meta"], meta)


def clear_analysis_cache(student_id: str) -> None:
    paths = get_student_file_paths(student_id)
    if os.path.exists(paths["analysis"]):
        os.remove(paths["analysis"])