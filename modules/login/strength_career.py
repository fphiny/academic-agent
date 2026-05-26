import os
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


GRADE_SCORE_MAP = {
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


PROJECT_KEYWORDS = [
    "프로젝트",
    "캡스톤",
    "실습",
    "설계",
    "응용",
    "특강",
    "세미나",
    "스튜디오",
    "연구",
    "워크숍",
]


class StrengthDraftItem(BaseModel):
    domain: str = Field(description="학생의 강점 분야명")
    description: str = Field(description="강점 설명")
    top_subjects: List[str] = Field(description="대표 과목명 2~5개, 반드시 실제 입력 과목명")


class CareerItem(BaseModel):
    career: str = Field(description="추천 진로명")
    based_on: str = Field(description="근거가 된 강점 분야")
    reason: str = Field(description="추천 이유")
    strength_summary: str = Field(description="강점 요약")
    skills: List[str] = Field(description="핵심 역량 3~5개")
    recommended_actions: List[str] = Field(description="추천 액션 2~4개")


class AnalysisDraftResult(BaseModel):
    strengths: List[StrengthDraftItem] = Field(description="강점 최대 3개")
    career_recommendations: List[CareerItem] = Field(description="추천 진로 정확히 4개")


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalize_string_list(values: Any, min_items: int = 0, max_items: int = 5) -> List[str]:
    if not isinstance(values, list):
        return []

    cleaned: List[str] = []
    seen = set()

    for item in values:
        text = _safe_str(item)
        if not text:
            continue
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= max_items:
            break

    if len(cleaned) < min_items:
        return cleaned

    return cleaned


def _normalize_grade(value: Any) -> str:
    return _safe_str(value).upper()


def _compute_gpa(df: pd.DataFrame) -> Optional[float]:
    if df.empty or "성적" not in df.columns or "학점" not in df.columns:
        return None

    temp = df.copy()
    temp["성적"] = temp["성적"].astype(str).str.strip().str.upper()
    temp["학점"] = pd.to_numeric(temp["학점"], errors="coerce")
    temp["평점"] = temp["성적"].map(GRADE_SCORE_MAP)
    temp = temp.dropna(subset=["학점", "평점"])

    if temp.empty:
        return None

    total_credits = temp["학점"].sum()
    if total_credits <= 0:
        return None

    gpa = (temp["학점"] * temp["평점"]).sum() / total_credits
    return round(float(gpa), 2)


def _prepare_course_payload(df: pd.DataFrame, max_rows: int = 160) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    temp = df.copy()

    for col in ["년도", "학기", "과목명", "학점", "성적", "이수구분"]:
        if col not in temp.columns:
            temp[col] = None

    temp["년도_num"] = pd.to_numeric(temp["년도"], errors="coerce").fillna(0)
    temp["학기_num"] = pd.to_numeric(temp["학기"], errors="coerce").fillna(0)
    temp["학점_num"] = pd.to_numeric(temp["학점"], errors="coerce").fillna(0)
    temp["grade_score"] = (
        temp["성적"].astype(str).str.strip().str.upper().map(GRADE_SCORE_MAP).fillna(-1)
    )

    temp = temp.sort_values(
        by=["년도_num", "학기_num", "학점_num", "grade_score", "과목명"],
        ascending=[False, False, False, False, True],
    ).head(max_rows)

    rows: List[Dict[str, Any]] = []
    for _, row in temp.iterrows():
        rows.append(
            {
                "year": _safe_str(row.get("년도")),
                "semester": _safe_str(row.get("학기")),
                "course_name": _safe_str(row.get("과목명")),
                "credits": _safe_float(row.get("학점")),
                "grade": _safe_str(row.get("성적")).upper(),
                "category": _safe_str(row.get("이수구분")),
            }
        )
    return rows


def _build_prompt_payload(df: pd.DataFrame) -> Dict[str, Any]:
    temp = df.copy()

    if "학점" not in temp.columns:
        temp["학점"] = None
    temp["학점"] = pd.to_numeric(temp["학점"], errors="coerce").fillna(0.0)

    semester_count = 0
    if "년도" in temp.columns and "학기" in temp.columns:
        temp["학기키"] = temp["년도"].astype(str) + "-" + temp["학기"].astype(str)
        semester_count = int(temp["학기키"].nunique())

    payload = {
        "summary": {
            "total_courses": int(len(temp)),
            "total_credits": round(float(temp["학점"].sum()), 1),
            "gpa": _compute_gpa(temp),
            "semester_count": semester_count,
        },
        "courses": _prepare_course_payload(temp),
    }
    return payload


def _ensure_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    for col in ["년도", "학기", "과목명", "학점", "성적", "이수구분"]:
        if col not in temp.columns:
            temp[col] = None

    temp["과목명"] = temp["과목명"].astype(str).str.strip()
    temp["학점"] = pd.to_numeric(temp["학점"], errors="coerce").fillna(0.0)
    temp["성적"] = temp["성적"].astype(str).str.strip().str.upper()
    temp["grade_score"] = temp["성적"].map(GRADE_SCORE_MAP)
    temp["년도_num"] = pd.to_numeric(temp["년도"], errors="coerce")
    temp["학기_num"] = pd.to_numeric(temp["학기"], errors="coerce")

    temp["semester_key"] = (
        temp["년도_num"].fillna(-1).astype(int).astype(str)
        + "-"
        + temp["학기_num"].fillna(-1).astype(int).astype(str)
    )

    temp["is_project_like"] = temp["과목명"].apply(
        lambda x: any(keyword in x for keyword in PROJECT_KEYWORDS)
    )
    return temp


def _match_subjects_to_rows(df: pd.DataFrame, top_subjects: List[str]) -> pd.DataFrame:
    if df.empty or not top_subjects:
        return df.iloc[0:0].copy()

    normalized_targets = {_safe_str(s) for s in top_subjects if _safe_str(s)}
    if not normalized_targets:
        return df.iloc[0:0].copy()

    matched = df[df["과목명"].isin(normalized_targets)].copy()

    if matched.empty:
        return matched

    matched = matched.drop_duplicates(subset=["과목명"], keep="first")
    return matched


def _calc_avg_score(matched_df: pd.DataFrame) -> float:
    valid = matched_df.dropna(subset=["grade_score"]).copy()
    if valid.empty:
        return 0.0
    return round(float(valid["grade_score"].mean()), 2)


def _calc_credit_sum(matched_df: pd.DataFrame) -> float:
    if matched_df.empty:
        return 0.0
    return round(float(matched_df["학점"].sum()), 1)


def _calc_course_count(matched_df: pd.DataFrame) -> int:
    if matched_df.empty:
        return 0
    return int(len(matched_df))


def _calc_semester_span_ratio(matched_df: pd.DataFrame, total_semesters: int) -> float:
    if matched_df.empty or total_semesters <= 0:
        return 0.0
    semester_count = matched_df["semester_key"].nunique()
    return min(1.0, float(semester_count) / float(total_semesters))


def _calc_project_ratio(matched_df: pd.DataFrame) -> float:
    if matched_df.empty:
        return 0.0
    return float(matched_df["is_project_like"].mean())


def _calc_strength_score(
    matched_df: pd.DataFrame,
    full_df: pd.DataFrame,
    total_semesters: int,
) -> float:
    if matched_df.empty or full_df.empty:
        return 0.0

    total_credits_all = float(full_df["학점"].sum()) if "학점" in full_df.columns else 0.0
    matched_credits = float(matched_df["학점"].sum())
    credit_ratio = (matched_credits / total_credits_all) if total_credits_all > 0 else 0.0
    credit_ratio = min(1.0, max(0.0, credit_ratio))

    avg_score = _calc_avg_score(matched_df)
    grade_ratio = min(1.0, max(0.0, avg_score / 4.5))

    semester_span_ratio = _calc_semester_span_ratio(matched_df, total_semesters)
    project_ratio = _calc_project_ratio(matched_df)

    raw = (
        grade_ratio * 0.45
        + credit_ratio * 0.30
        + semester_span_ratio * 0.15
        + project_ratio * 0.10
    )

    return round(raw * 100.0, 1)


def _postprocess_strengths(
    grade_df: pd.DataFrame,
    strength_drafts: List[StrengthDraftItem],
) -> List[Dict[str, Any]]:
    df = _ensure_analysis_columns(grade_df)

    total_semesters = 0
    if not df.empty:
        total_semesters = int(df["semester_key"].nunique())

    results: List[Dict[str, Any]] = []
    used_domains = set()

    for item in strength_drafts:
        domain = _safe_str(item.domain)
        description = _safe_str(item.description)
        top_subjects = [_safe_str(s) for s in item.top_subjects if _safe_str(s)]

        if not domain or domain in used_domains:
            continue

        matched_df = _match_subjects_to_rows(df, top_subjects)
        if matched_df.empty:
            continue

        actual_subjects = matched_df["과목명"].dropna().astype(str).tolist()[:5]
        avg_score = _calc_avg_score(matched_df)
        credit_sum = _calc_credit_sum(matched_df)
        course_count = _calc_course_count(matched_df)
        score = _calc_strength_score(matched_df, df, total_semesters)

        results.append(
            {
                "domain": domain,
                "score": max(0.0, min(100.0, score)),
                "credit_sum": max(0.0, credit_sum),
                "course_count": max(0, course_count),
                "avg_score": max(0.0, min(4.5, avg_score)),
                "description": description,
                "top_subjects": actual_subjects,
            }
        )
        used_domains.add(domain)

        if len(results) >= 3:
            break

    results.sort(key=lambda x: (x["score"], x["avg_score"], x["credit_sum"]), reverse=True)
    return results[:3]


def _postprocess_careers(
    careers: List[CareerItem],
    strengths: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not strengths:
        return []

    valid_domains = {s["domain"] for s in strengths}
    cleaned: List[Dict[str, Any]] = []
    seen_careers = set()

    for item in careers:
        career = _safe_str(item.career)
        based_on = _safe_str(item.based_on)
        reason = _safe_str(item.reason)
        strength_summary = _safe_str(item.strength_summary)
        skills = _normalize_string_list(item.skills, min_items=0, max_items=5)
        recommended_actions = _normalize_string_list(item.recommended_actions, min_items=0, max_items=4)

        if not career:
            continue
        if career in seen_careers:
            continue

        if based_on not in valid_domains:
            based_on = strengths[0]["domain"]

        if not reason:
            reason = strength_summary or f"{based_on} 강점을 바탕으로 추천되는 진로입니다."

        if not strength_summary:
            strength_summary = f"{based_on} 관련 학업 성취와 수강 흐름이 확인됩니다."

        cleaned.append(
            {
                # 기존 키
                "career": career,
                "based_on": based_on,
                "reason": reason,
                "strength_summary": strength_summary,
                "skills": skills,
                "recommended_actions": recommended_actions,

                # 프론트 호환 키
                "career_name": career,
                "description": reason,
            }
        )
        seen_careers.add(career)

        if len(cleaned) >= 4:
            break

    if len(cleaned) != 4:
        raise RuntimeError(
            f"OpenAI output must contain exactly 4 career recommendations, got {len(cleaned)}."
        )

    return cleaned


def analyze_strengths_and_careers(grade_df: pd.DataFrame) -> dict:
    """
    강점/진로 분석
    - LLM: 강점 도메인, 설명, 대표 과목, 진로 추천 생성
    - 코드: credit_sum, course_count, avg_score, score 계산
    """
    if grade_df is None or grade_df.empty:
        return {
            "strengths": [],
            "career_recommendations": [],
        }

    client = _get_client()
    payload = _build_prompt_payload(grade_df)

    system_prompt = """
너는 한국 대학생의 성적표를 보고 강점과 진로를 분석하는 전문 커리어 분석기다.

반드시 지켜라:
1. 과목명, 학점, 성적, 이수량의 패턴만 보고 판단한다.
2. 수동 규칙처럼 보이는 고정 직무 추천을 하지 말고, 입력 데이터에 맞춰 추론하라.
3. strengths는 최대 3개만 출력한다.
4. career_recommendations는 정확히 4개 출력한다.
5. 강점 설명과 진로 추천은 반드시 한국어로 작성한다.
6. 근거 없는 과장 금지. 애매하면 보수적으로 표현한다.
7. top_subjects에는 실제 입력 과목명만 넣는다.
8. domain은 학생 데이터에서 추론한 분야명으로 유연하게 만든다.
   예: 백엔드/서버개발, 데이터/AI, 프론트엔드, 보안/인프라, 임베디드, 게임개발, 모바일, 기획/PM, 연구형 등
9. avg_score, credit_sum, course_count, score는 출력하지 마라. 이 값들은 시스템이 별도 계산한다.
10. strengths의 각 항목에는 반드시 domain, description, top_subjects만 넣어라.
11. top_subjects는 각 강점마다 2~5개 작성하라.
12. top_subjects에는 반드시 실제 입력 과목명만 사용하라.
13. career 추천은 강점 1~3위에서 직접 파생되어야 하며 4개는 최대한 덜 겹치게 작성하라.
14. career_recommendations의 각 항목에는 반드시
    career, based_on, reason, strength_summary, skills, recommended_actions만 넣어라.
15. skills는 각 진로마다 3~5개 작성하라.
16. recommended_actions는 각 진로마다 2~4개 작성하라.
17. skills와 recommended_actions는 반드시 한국어로 작성하라.
18. recommended_actions는 학생이 실제로 바로 실천할 수 있는 행동으로 작성하라.
19. 결과는 스키마에 맞게만 출력한다.

판단 기준:
- 과목명에서 드러나는 전공 분야
- 특정 분야 과목의 누적 수강량과 학점
- 해당 분야 과목의 성적 수준
- 프로젝트, 실습, 캡스톤 과목의 존재
- 여러 학기에 걸친 일관성
- 서로 연결되는 과목 조합
""".strip()

    few_shot_user_1 = {
        "summary": {
            "total_courses": 10,
            "total_credits": 30.0,
            "gpa": 4.02,
            "semester_count": 2,
        },
        "courses": [
            {"year": "2023", "semester": "1", "course_name": "자료구조", "credits": 3.0, "grade": "A+", "category": "전공필수"},
            {"year": "2023", "semester": "1", "course_name": "객체지향프로그래밍", "credits": 3.0, "grade": "A0", "category": "전공필수"},
            {"year": "2023", "semester": "1", "course_name": "컴퓨터구조", "credits": 3.0, "grade": "B+", "category": "전공필수"},
            {"year": "2023", "semester": "2", "course_name": "운영체제", "credits": 3.0, "grade": "A0", "category": "전공필수"},
            {"year": "2023", "semester": "2", "course_name": "데이터베이스", "credits": 3.0, "grade": "A+", "category": "전공필수"},
            {"year": "2023", "semester": "2", "course_name": "네트워크", "credits": 3.0, "grade": "A0", "category": "전공선택"},
            {"year": "2023", "semester": "2", "course_name": "웹프로그래밍", "credits": 3.0, "grade": "A0", "category": "전공선택"},
        ],
    }

    few_shot_assistant_1 = {
        "strengths": [
            {
                "domain": "백엔드/서버개발",
                "description": "자료구조, 운영체제, 데이터베이스, 네트워크, 웹프로그래밍 계열 과목에서 높은 성취를 보여 서버 구조와 데이터 처리 중심의 개발 역량이 강점으로 보입니다.",
                "top_subjects": ["자료구조", "운영체제", "데이터베이스", "네트워크", "웹프로그래밍"],
            },
            {
                "domain": "소프트웨어 설계",
                "description": "객체지향프로그래밍과 컴퓨터구조 등 기초 설계·구현 과목 이수가 좋아 안정적인 개발 기반을 갖춘 편입니다.",
                "top_subjects": ["객체지향프로그래밍", "컴퓨터구조", "자료구조"],
            },
        ],
        "career_recommendations": [
            {
                "career": "백엔드 개발자",
                "based_on": "백엔드/서버개발",
                "reason": "서버, 데이터베이스, 네트워크 관련 과목 조합과 성적 흐름이 가장 직접적으로 연결됩니다.",
                "strength_summary": "서버 구조 이해와 데이터 처리 중심의 개발 역량이 확인됩니다.",
                "skills": ["서버 로직 구현", "데이터베이스 설계", "API 설계", "문제 해결력"],
                "recommended_actions": ["백엔드 프로젝트 1개 완성", "SQL 심화 학습", "REST API 포트폴리오 정리"],
            },
            {
                "career": "플랫폼 엔지니어",
                "based_on": "백엔드/서버개발",
                "reason": "운영체제·네트워크·데이터베이스 기반이 있어 서비스 운영 관점의 시스템 개발과 잘 맞습니다.",
                "strength_summary": "시스템 기초와 서비스 백엔드 역량이 함께 보입니다.",
                "skills": ["리눅스 기초", "네트워크 이해", "시스템 문제 해결", "서비스 운영 관점"],
                "recommended_actions": ["리눅스 서버 실습", "네트워크 기초 복습", "간단한 배포 경험 쌓기"],
            },
            {
                "career": "서버 사이드 소프트웨어 엔지니어",
                "based_on": "소프트웨어 설계",
                "reason": "객체지향 및 기초 전공 기반이 안정적이라 서비스 로직 구현 직무로 이어지기 좋습니다.",
                "strength_summary": "설계 기반 개발 역량이 비교적 탄탄합니다.",
                "skills": ["객체지향 설계", "코드 구조화", "로직 구현", "협업 개발 기초"],
                "recommended_actions": ["객체지향 리팩토링 연습", "토이 프로젝트 구조 개선", "Git 협업 흐름 익히기"],
            },
            {
                "career": "웹 서비스 개발자",
                "based_on": "백엔드/서버개발",
                "reason": "웹프로그래밍과 데이터 처리 과목의 조합이 웹 서비스 구현 직무와 잘 연결됩니다.",
                "strength_summary": "서비스 기능 구현과 데이터 처리 흐름을 이해하는 기반이 있습니다.",
                "skills": ["웹 애플리케이션 이해", "API 연동", "데이터 처리", "서비스 구현"],
                "recommended_actions": ["웹 서비스 포트폴리오 제작", "간단한 로그인 기능 구현", "DB 연동 프로젝트 완성"],
            },
        ],
    }

    few_shot_user_2 = {
        "summary": {
            "total_courses": 9,
            "total_credits": 27.0,
            "gpa": 3.88,
            "semester_count": 2,
        },
        "courses": [
            {"year": "2024", "semester": "1", "course_name": "선형대수", "credits": 3.0, "grade": "A0", "category": "전공기초"},
            {"year": "2024", "semester": "1", "course_name": "확률통계", "credits": 3.0, "grade": "A+", "category": "전공기초"},
            {"year": "2024", "semester": "1", "course_name": "파이썬프로그래밍", "credits": 3.0, "grade": "A0", "category": "전공선택"},
            {"year": "2024", "semester": "2", "course_name": "데이터마이닝", "credits": 3.0, "grade": "A0", "category": "전공선택"},
            {"year": "2024", "semester": "2", "course_name": "머신러닝", "credits": 3.0, "grade": "A0", "category": "전공선택"},
            {"year": "2024", "semester": "2", "course_name": "딥러닝", "credits": 3.0, "grade": "B+", "category": "전공선택"},
            {"year": "2024", "semester": "2", "course_name": "빅데이터분석", "credits": 3.0, "grade": "A+", "category": "전공선택"},
        ],
    }

    few_shot_assistant_2 = {
        "strengths": [
            {
                "domain": "데이터/AI",
                "description": "통계·수학 기반 과목과 머신러닝·딥러닝·데이터마이닝 과목이 자연스럽게 연결되어 있어 데이터 분석과 모델링 역량이 강한 편입니다.",
                "top_subjects": ["확률통계", "선형대수", "데이터마이닝", "머신러닝", "빅데이터분석"],
            }
        ],
        "career_recommendations": [
            {
                "career": "데이터 분석가",
                "based_on": "데이터/AI",
                "reason": "통계와 데이터 처리 과목 조합이 분석 직무와 직접 연결됩니다.",
                "strength_summary": "데이터 해석과 분석 기반 문제 해결 역량이 강합니다.",
                "skills": ["데이터 해석", "통계적 사고", "시각화 기초", "분석 문제 해결"],
                "recommended_actions": ["분석 프로젝트 1개 정리", "통계 복습", "대시보드 제작 연습"],
            },
            {
                "career": "머신러닝 엔지니어",
                "based_on": "데이터/AI",
                "reason": "머신러닝·딥러닝 과목 이수와 성적이 확인되어 모델 개발 직무 적합성이 높습니다.",
                "strength_summary": "분석을 넘어 모델링까지 이어지는 학업 패턴이 보입니다.",
                "skills": ["모델링 기초", "파이썬 활용", "데이터 전처리", "실험 반복 개선"],
                "recommended_actions": ["ML 토이 프로젝트 구현", "모델 성능 비교 실험", "전처리 코드 정리"],
            },
            {
                "career": "AI 응용 개발자",
                "based_on": "데이터/AI",
                "reason": "파이썬과 AI 과목 조합이 있어 실제 응용 서비스 구현으로 확장하기 좋습니다.",
                "strength_summary": "데이터와 모델을 실제 개발로 연결할 수 있는 기반이 있습니다.",
                "skills": ["파이썬 개발", "모델 활용", "서비스 응용", "문제 해결"],
                "recommended_actions": ["AI 기능 포함 앱 구현", "오픈소스 모델 연동", "포트폴리오 데모 제작"],
            },
            {
                "career": "데이터 사이언티스트",
                "based_on": "데이터/AI",
                "reason": "수학·통계 기반과 데이터 분석 과목 흐름이 함께 있어 탐색적 분석과 모델링 모두 적합합니다.",
                "strength_summary": "분석과 모델링을 함께 수행할 수 있는 학업 기반이 보입니다.",
                "skills": ["통계 모델링", "가설 검증", "데이터 탐색", "문제 구조화"],
                "recommended_actions": ["EDA 보고서 작성", "회귀/분류 실험 정리", "분석 노트북 포트폴리오화"],
            },
        ],
    }

    input_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "예시 입력 1\n" + json.dumps(few_shot_user_1, ensure_ascii=False, indent=2),
        },
        {
            "role": "assistant",
            "content": json.dumps(few_shot_assistant_1, ensure_ascii=False, indent=2),
        },
        {
            "role": "user",
            "content": "예시 입력 2\n" + json.dumps(few_shot_user_2, ensure_ascii=False, indent=2),
        },
        {
            "role": "assistant",
            "content": json.dumps(few_shot_assistant_2, ensure_ascii=False, indent=2),
        },
        {
            "role": "user",
            "content": "실제 입력\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=input_messages,
        text_format=AnalysisDraftResult,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed output.")

    strengths = _postprocess_strengths(grade_df, parsed.strengths)
    careers = _postprocess_careers(parsed.career_recommendations, strengths)

    return {
        "strengths": strengths,
        "career_recommendations": careers,
    }