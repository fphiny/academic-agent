def analyze_student(grade_df: pd.DataFrame) -> dict:
    return {
        "total_courses": len(grade_df),
        "semesters": sorted(grade_df["학기키"].dropna().unique().tolist()),
        "average_score": None,
        "retake_courses": [],
        "low_grades": [],
    }