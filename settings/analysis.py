from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.login.strength_career import analyze_strengths_and_careers
from settings.storage import load_cached_analysis, save_analysis_cache


def calculate_gpa(grade_df: pd.DataFrame) -> Optional[float]:
    if grade_df.empty or "성적" not in grade_df.columns or "학점" not in grade_df.columns:
        return None

    grade_map = {
        "A+": 4.5,
        "A0": 4.0,
        "B+": 3.5,
        "B0": 3.0,
        "C+": 2.5,
        "C0": 2.0,
        "D+": 1.5,
        "D0": 1.0,
        "F": 0.0,
    }

    df = grade_df.copy()
    df["성적"] = df["성적"].astype(str).str.strip().str.upper()
    df["학점"] = pd.to_numeric(df["학점"], errors="coerce")
    df["평점"] = df["성적"].map(grade_map)
    df = df.dropna(subset=["학점", "평점"])

    if df.empty:
        return None

    total_credits = df["학점"].sum()
    if total_credits == 0:
        return None

    gpa = (df["학점"] * df["평점"]).sum() / total_credits
    return round(float(gpa), 2)


def analyze_grades(grade_df: pd.DataFrame, student_id: Optional[str] = None) -> dict:
    if grade_df.empty:
        return {
            "summary": {
                "total_courses": 0,
                "total_credits": 0,
                "gpa": None,
                "semester_count": 0,
            },
            "semesters": [],
            "low_grade_courses": [],
            "retake_candidates": [],
            "category_credit_summary": {},
            "strengths": [],
            "career_recommendations": [],
        }

    df = grade_df.copy()

    for col in ["년도", "학기", "학기키", "과목명", "학점", "성적", "이수구분"]:
        if col not in df.columns:
            df[col] = None

    df["학점"] = pd.to_numeric(df["학점"], errors="coerce")
    df["학기키"] = df["학기키"].fillna(df["년도"].astype(str) + "-" + df["학기"].astype(str))

    total_credits = float(df["학점"].fillna(0).sum())
    gpa = calculate_gpa(df)

    semester_stats = []
    for semester_key, group in df.groupby("학기키", sort=True):
        semester_gpa = calculate_gpa(group)
        semester_stats.append(
            {
                "semester": semester_key,
                "course_count": int(len(group)),
                "credit_sum": float(group["학점"].fillna(0).sum()),
                "gpa": semester_gpa,
                "courses": group[["과목명", "학점", "성적", "이수구분"]]
                .fillna("")
                .to_dict(orient="records"),
            }
        )

    grade_score_map = {
        "A+": 4.5,
        "A0": 4.0,
        "B+": 3.5,
        "B0": 3.0,
        "C+": 2.5,
        "C0": 2.0,
        "D+": 1.5,
        "D0": 1.0,
        "F": 0.0,
    }

    df["성적점수"] = df["성적"].astype(str).str.strip().str.upper().map(grade_score_map)

    low_grade_df = df[df["성적점수"].notna() & (df["성적점수"] <= 2.5)].copy()
    low_grade_courses = (
        low_grade_df[["학기키", "과목명", "학점", "성적", "이수구분"]]
        .fillna("")
        .to_dict(orient="records")
    )

    repeated_name_counts = df["과목명"].value_counts()
    repeated_course_names = repeated_name_counts[repeated_name_counts > 1].index.tolist()
    retake_candidates_df = df[df["과목명"].isin(repeated_course_names)].copy()
    retake_candidates = (
        retake_candidates_df[["학기키", "과목명", "학점", "성적", "이수구분"]]
        .fillna("")
        .to_dict(orient="records")
    )

    category_credit_summary = {}
    if "이수구분" in df.columns:
        category_group = df.groupby("이수구분", dropna=False)["학점"].sum()
        category_credit_summary = {
            str(k) if k is not None else "기타": float(v)
            for k, v in category_group.items()
        }

    strength_result = None

    if student_id:
        strength_result = load_cached_analysis(student_id)

    if strength_result is None:
        strength_result = analyze_strengths_and_careers(df)
        if student_id:
            save_analysis_cache(student_id, strength_result)

    return {
        "summary": {
            "total_courses": int(len(df)),
            "total_credits": round(total_credits, 1),
            "gpa": gpa,
            "semester_count": int(df["학기키"].nunique()),
        },
        "semesters": semester_stats,
        "low_grade_courses": low_grade_courses,
        "retake_candidates": retake_candidates,
        "category_credit_summary": category_credit_summary,
        "strengths": strength_result.get("strengths", []),
        "career_recommendations": strength_result.get("career_recommendations", []),
    }