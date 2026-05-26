# modules/login/route.py
from __future__ import annotations

import csv
import os
import uuid
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from modules.chat.recommendations import recommend_courses_for_career_question
from modules.login.login_crawler import crawl_student_data
from modules.login.strength_career import analyze_strengths_and_careers
from settings.config import TEMPLATES_DIR
from settings.schemas import CourseRecommendationRequest, LoginRequest
from settings.storage import (
    GRADE_STORE,
    dataframe_to_session_records,
    load_cached_analysis,
    load_student_grades,
    save_analysis_cache,
    save_student_grades,
    session_records_to_dataframe,
)

router = APIRouter(tags=["login"])
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# demo accounts
DEMO_ACCOUNTS = {
    "demo_yuko": "demo1234",
    "demo_hallym1": "demo5678",
    "demo_hallym2": "demo5678",
    "user_bis": "password_bis",
}

# DB 접근 권한 관리자 CSV
ADMIN_USERS_CSV = os.path.join(os.path.dirname(__file__), "admin_users.csv")


def load_admin_student_ids() -> Set[str]:
    """
    modules/login/admin_users.csv 파일에서 DB 접근 가능 student_id 목록을 읽는다.

    CSV 예시:
    student_id
    user_bis
    demo_yuko
    20201234
    """
    if not os.path.exists(ADMIN_USERS_CSV):
        return set()

    admin_ids: Set[str] = set()

    with open(ADMIN_USERS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            student_id = str(row.get("student_id", "")).strip()
            if student_id:
                admin_ids.add(student_id)

    return admin_ids


def can_access_db_for_student(student_id: str) -> bool:
    return student_id.strip() in load_admin_student_ids()


# -------------------------------------------------------
# session helpers
# -------------------------------------------------------


def get_session(request: Request) -> Dict[Any, Any]:
    return request.session


def get_logged_in_store_key(session: Dict[Any, Any]) -> Optional[str]:
    store_key = session.get("grade_store_key")
    if not store_key:
        return None
    if store_key not in GRADE_STORE:
        return None
    return store_key


def get_logged_in_student_id(session: Dict[Any, Any]) -> Optional[str]:
    store_key = get_logged_in_store_key(session)
    if not store_key:
        return None
    return GRADE_STORE[store_key].get("student_id")


def get_can_access_db(session: Dict[Any, Any]) -> bool:
    return bool(session.get("can_access_db"))


async def check_login_status(
    session: Dict[Any, Any] = Depends(get_session),
) -> Optional[str]:
    return get_logged_in_student_id(session)


# -------------------------------------------------------
# grade analysis
# -------------------------------------------------------


def calculate_gpa(grade_df) -> Optional[float]:
    import pandas as pd

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


def analyze_grades(grade_df, student_id: Optional[str] = None) -> dict:
    import pandas as pd

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


# -------------------------------------------------------
# pages
# -------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    student_id: Optional[str] = Depends(check_login_status),
):
    if student_id:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

    login_template_path = os.path.join(TEMPLATES_DIR, "login.html")
    if os.path.exists(login_template_path):
        return templates.TemplateResponse("login.html", {"request": request})

    return HTMLResponse(
        """
        <html>
            <body>
                <h2>학생 로그인</h2>
                <p>POST /login 으로 student_id, password JSON 전송</p>
                <p><a href="/rag">RAG 화면</a></p>
            </body>
        </html>
        """
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    session: Dict[Any, Any] = Depends(get_session),
):
    store_key = get_logged_in_store_key(session)
    if not store_key:
        session.clear()
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

    student_id = GRADE_STORE[store_key]["student_id"]
    grade_df = session_records_to_dataframe(GRADE_STORE[store_key]["grades"])

    if grade_df.empty:
        grade_df = load_student_grades(student_id)

    analysis = analyze_grades(grade_df, student_id=student_id)

    dashboard_template_path = os.path.join(TEMPLATES_DIR, "dashboard.html")
    if os.path.exists(dashboard_template_path):
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "student_id": student_id,
                "analysis": analysis,
                "grades": grade_df.fillna("").to_dict(orient="records"),
                "can_access_db": get_can_access_db(session),
            },
        )

    return HTMLResponse(
        f"""
        <html>
            <body>
                <h2>로그인 완료</h2>
                <p>student_id: {student_id}</p>
                <p>DB 접속 권한: {get_can_access_db(session)}</p>
                <p>총 과목 수: {analysis["summary"]["total_courses"]}</p>
                <p>총 이수학점: {analysis["summary"]["total_credits"]}</p>
                <p>평점(GPA): {analysis["summary"]["gpa"]}</p>
                <p><a href="/grades">성적 JSON 보기</a></p>
                <p><a href="/analyze">분석 JSON 보기</a></p>
                <p><a href="/rag">RAG 화면</a></p>
                <p><a href="/logout">로그아웃</a></p>
            </body>
        </html>
        """
    )


# -------------------------------------------------------
# student auth / grade APIs
# -------------------------------------------------------


@router.post("/login")
async def login(
    login_data: LoginRequest,
    session: Dict[Any, Any] = Depends(get_session),
):
    student_id = login_data.student_id.strip()
    password = login_data.password

    if not student_id or not password:
        raise HTTPException(status_code=400, detail="학번과 비밀번호를 입력하세요.")

    print(f"🚀 [Login] 시도: {student_id}")

    # -------------------------------------------------------
    # demo login
    # -------------------------------------------------------
    demo_password = DEMO_ACCOUNTS.get(student_id)
    if demo_password is not None:
        if password != demo_password:
            raise HTTPException(status_code=401, detail="데모 계정 비밀번호가 올바르지 않습니다.")

        old_store_key = session.get("grade_store_key")
        if old_store_key and old_store_key in GRADE_STORE:
            del GRADE_STORE[old_store_key]

        store_key = str(uuid.uuid4())

        GRADE_STORE[store_key] = {
            "student_id": student_id,
            "grades": [],
        }

        can_access_db = can_access_db_for_student(student_id)

        session["student_id"] = student_id
        session["grade_store_key"] = store_key
        session["can_access_db"] = can_access_db

        print(
            f"✅ [Demo Login] 성공: {student_id}, "
            f"store_key={store_key}, can_access_db={can_access_db}"
        )

        return JSONResponse(
            content={
                "success": True,
                "student_id": student_id,
                "row_count": 0,
                "redirect": "/rag",
                "is_demo": True,
                "can_access_db": can_access_db,
            }
        )

    try:
        grade_df = await run_in_threadpool(crawl_student_data, student_id, password)

        if grade_df is None:
            raise HTTPException(status_code=401, detail="로그인 실패")

        old_store_key = session.get("grade_store_key")
        if old_store_key and old_store_key in GRADE_STORE:
            del GRADE_STORE[old_store_key]

        store_key = str(uuid.uuid4())
        grade_records = dataframe_to_session_records(grade_df)

        GRADE_STORE[store_key] = {
            "student_id": student_id,
            "grades": grade_records,
        }

        save_student_grades(student_id, grade_df)

        can_access_db = can_access_db_for_student(student_id)

        session["student_id"] = student_id
        session["grade_store_key"] = store_key
        session["can_access_db"] = can_access_db

        print(
            f"✅ [Login] 성공: {student_id}, "
            f"rows={len(grade_df)}, store_key={store_key}, can_access_db={can_access_db}"
        )

        return JSONResponse(
            content={
                "success": True,
                "student_id": student_id,
                "row_count": int(len(grade_df)),
                "redirect": "/dashboard",
                "can_access_db": can_access_db,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [Login Error] {e}")
        raise HTTPException(status_code=500, detail="로그인 중 오류가 발생했습니다.")


@router.get("/grades")
async def get_grades(
    session: Dict[Any, Any] = Depends(get_session),
):
    store_key = get_logged_in_store_key(session)
    if not store_key:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    student_id = GRADE_STORE[store_key]["student_id"]
    grade_df = session_records_to_dataframe(GRADE_STORE[store_key]["grades"])

    if grade_df.empty:
        grade_df = load_student_grades(student_id)

    return JSONResponse(
        content={
            "success": True,
            "student_id": student_id,
            "row_count": int(len(grade_df)),
            "grades": dataframe_to_session_records(grade_df),
        }
    )


@router.get("/analyze")
async def analyze(
    session: Dict[Any, Any] = Depends(get_session),
):
    store_key = get_logged_in_store_key(session)
    if not store_key:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    student_id = GRADE_STORE[store_key]["student_id"]
    grade_df = session_records_to_dataframe(GRADE_STORE[store_key]["grades"])

    if grade_df.empty:
        grade_df = load_student_grades(student_id)

    result = analyze_grades(grade_df, student_id=student_id)

    return JSONResponse(
        content={
            "success": True,
            "student_id": student_id,
            "analysis": result,
        }
    )


@router.get("/logout")
async def logout(
    session: Dict[Any, Any] = Depends(get_session),
):
    store_key = session.get("grade_store_key")
    if store_key and store_key in GRADE_STORE:
        del GRADE_STORE[store_key]

    session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)