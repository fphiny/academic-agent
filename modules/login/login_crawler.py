import time
import re
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (
    TimeoutException,
    UnexpectedAlertPresentException,
    NoSuchElementException,
    WebDriverException,
)


# =========================
# 유틸
# =========================
def safe_strip(value) -> str:
    return str(value).strip() if value is not None else ""


def semester_key(year: str, semester: str) -> str:
    return f"{year}-{semester}"


def normalize_grade(grade: str) -> str:
    """
    학교 시스템에서 A0를 Ao로 주는 경우가 있어 보정
    """
    g = safe_strip(grade)
    mapping = {
        "Ao": "A0",
        "Bo": "B0",
        "Co": "C0",
        "Do": "D0",
        "Fo": "F0",
    }
    return mapping.get(g, g)


def parse_summary_text(text: str) -> dict:
    """
    예:
    신청학점 : 14    취득학점 : 14    평점합계 : 43.5    평점평균 :  4.35
    """
    text = safe_strip(text)

    def extract(pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""

    return {
        "신청학점": extract(r"신청학점\s*:\s*([0-9.]+)"),
        "취득학점": extract(r"취득학점\s*:\s*([0-9.]+)"),
        "평점합계": extract(r"평점합계\s*:\s*([0-9.]+)"),
        "평점평균": extract(r"평점평균\s*:\s*([0-9.]+)"),
    }


# =========================
# 파싱
# =========================
def parse_html_table(html_content: str) -> tuple[list[dict], list[dict]]:
    """
    HTML <tr><td> 구조 파싱
    GBN=1: 과목 상세
    GBN=2: 학기 요약
    """
    subject_rows = []
    summary_rows = []

    soup = BeautifulSoup(html_content, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return subject_rows, summary_rows

    for row in rows:
        cols = [safe_strip(td.get_text()) for td in row.find_all("td")]
        if len(cols) < 3:
            continue

        year = cols[0] if len(cols) > 0 else ""
        semester = cols[1] if len(cols) > 1 else ""
        gbn = cols[2] if len(cols) > 2 else ""

        if not (year.isdigit() and len(year) == 4):
            continue

        # 상세 과목행
        if gbn == "1" and len(cols) >= 13:
            subject_rows.append({
                "년도": year,
                "학기": semester,
                "학기키": semester_key(year, semester),
                "이수구분코드": cols[5] if len(cols) > 5 else "",
                "이수구분": cols[6] if len(cols) > 6 else "",
                "과목코드": cols[7] if len(cols) > 7 else "",
                "과목코드분반": cols[8] if len(cols) > 8 else "",
                "분반": cols[9] if len(cols) > 9 else "",
                "과목명": cols[10] if len(cols) > 10 else "",
                "학점": cols[11] if len(cols) > 11 else "",
                "성적": normalize_grade(cols[12] if len(cols) > 12 else ""),
                "재이수": cols[13] if len(cols) > 13 else "",
                "교수번호": cols[14] if len(cols) > 14 else "",
                "교수명": cols[15] if len(cols) > 15 else "",
            })

        # 학기 요약행
        elif gbn == "2":
            summary_text = " ".join(c for c in cols[4:] if c)
            parsed = parse_summary_text(summary_text)
            summary_rows.append({
                "년도": year,
                "학기": semester,
                "학기키": semester_key(year, semester),
                "요약원문": summary_text,
                **parsed,
            })

    return subject_rows, summary_rows


def parse_raw_records(text: str) -> tuple[list[dict], list[dict]]:
    """
    raw 응답 파싱
    레코드 구분: \x0c
    필드 구분: \x08
    """
    subject_rows = []
    summary_rows = []

    records = text.split("\x0c")

    for record in records:
        record = record.strip()
        if not record:
            continue

        cols = [safe_strip(c) for c in record.split("\x08")]

        if len(cols) < 3:
            continue

        year = cols[0]
        semester = cols[1]
        gbn = cols[2]

        if year.lower() == "tot":
            continue

        if year == "SUNG_YY":
            continue

        if not (year.isdigit() and len(year) == 4):
            continue

        # 상세 과목행
        if gbn == "1" and len(cols) >= 13:
            subject_rows.append({
                "년도": year,
                "학기": semester,
                "학기키": semester_key(year, semester),
                "학번": cols[3] if len(cols) > 3 else "",
                "이수구분코드": cols[5] if len(cols) > 5 else "",
                "이수구분": cols[6] if len(cols) > 6 else "",
                "과목코드": cols[7] if len(cols) > 7 else "",
                "과목코드분반": cols[8] if len(cols) > 8 else "",
                "분반": cols[9] if len(cols) > 9 else "",
                "과목명": cols[10] if len(cols) > 10 else "",
                "학점": cols[11] if len(cols) > 11 else "",
                "성적": normalize_grade(cols[12] if len(cols) > 12 else ""),
                "재이수": cols[13] if len(cols) > 13 else "",
                "교수번호": cols[14] if len(cols) > 14 else "",
                "교수명": cols[15] if len(cols) > 15 else "",
            })

        # 학기 요약행
        elif gbn == "2":
            summary_text_candidates = [c for c in cols[4:] if c]
            summary_text = " ".join(summary_text_candidates)
            parsed = parse_summary_text(summary_text)

            summary_rows.append({
                "년도": year,
                "학기": semester,
                "학기키": semester_key(year, semester),
                "학번": cols[3] if len(cols) > 3 else "",
                "요약원문": summary_text,
                **parsed,
            })

    return subject_rows, summary_rows


def parse_to_dataframe(raw_text: str) -> pd.DataFrame:
    """
    응답 전체를 파싱하여 과목 상세 DataFrame만 반환
    """
    subject_rows = []
    summary_rows = []

    # 1) HTML 테이블 파싱 시도
    try:
        html_subjects, html_summaries = parse_html_table(raw_text)
        subject_rows.extend(html_subjects)
        summary_rows.extend(html_summaries)
    except Exception as e:
        print(f"⚠️ HTML 파싱 실패: {e}")

    # 2) HTML 파싱 결과가 부실하면 raw record 파싱
    if not subject_rows:
        try:
            raw_subjects, raw_summaries = parse_raw_records(raw_text)
            subject_rows.extend(raw_subjects)
            summary_rows.extend(raw_summaries)
        except Exception as e:
            print(f"⚠️ Raw 파싱 실패: {e}")

    subject_df = pd.DataFrame(subject_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not subject_df.empty:
        subject_df["년도_num"] = pd.to_numeric(subject_df["년도"], errors="coerce")
        subject_df["학기_num"] = pd.to_numeric(subject_df["학기"], errors="coerce")
        subject_df = subject_df.sort_values(
            by=["년도_num", "학기_num", "과목명"],
            ascending=[False, False, True]
        ).drop(columns=["년도_num", "학기_num"]).reset_index(drop=True)

    if not summary_df.empty:
        print(f"✅ 학기 요약 추출 성공: {len(summary_df)}건")
    else:
        print("⚠️ 학기 요약 없음")

    if not subject_df.empty:
        print(f"✅ 과목 데이터 추출 성공: {len(subject_df)}건")
        return subject_df
    else:
        print("⚠️ 과목 데이터 없음")
        return pd.DataFrame(columns=[
            "년도", "학기", "학기키", "학번", "이수구분코드", "이수구분",
            "과목코드", "과목코드분반", "분반", "과목명", "학점", "성적",
            "재이수", "교수번호", "교수명"
        ])


# =========================
# Selenium 보조
# =========================
def handle_alert(driver) -> Optional[str]:
    """
    alert가 있으면 닫고 텍스트 반환
    없으면 None
    """
    try:
        WebDriverWait(driver, 0.7).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        text = alert.text
        print(f"⚠️ 팝업 발견: {text}")
        alert.accept()
        time.sleep(0.5)
        return text
    except Exception:
        return None


def wait_and_click(driver, wait: WebDriverWait, by, locator, desc: str = "") -> bool:
    try:
        element = wait.until(EC.element_to_be_clickable((by, locator)))
        element.click()
        return True
    except Exception as e:
        print(f"⚠️ 클릭 실패 [{desc or locator}]: {e}")
        return False


def login_and_navigate(driver, wait: WebDriverWait, student_id: str, password: str) -> bool:
    print(f"1) [{student_id}] 로그인 시도...")
    driver.get("https://was1.hallym.ac.kr:8087/hlwc/mdi/Login.html")
    handle_alert(driver)

    try:
        id_input = wait.until(EC.element_to_be_clickable((By.ID, "Form_login.id")))
        pw_input = wait.until(EC.element_to_be_clickable((By.ID, "Form_login.passwd")))
    except TimeoutException:
        print("❌ 로그인 페이지 요소를 찾지 못했습니다.")
        return False

    id_input.clear()
    id_input.send_keys(student_id)

    pw_input.clear()
    pw_input.send_keys(password)

    try:
        login_btn = driver.find_element(By.ID, "Form_login.send")
        login_btn.click()
    except NoSuchElementException:
        print("❌ 로그인 버튼을 찾지 못했습니다.")
        return False

    time.sleep(1.2)

    alert_text = handle_alert(driver)
    if alert_text:
        print(f"❌ 로그인 후 팝업 발생: {alert_text}")
        return False

    print("2) 메뉴 이동 중...")
    try:
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "LeftFrame")))
    except TimeoutException:
        print("❌ LeftFrame 진입 실패")
        return False

    menu_ids = [
        "left_Menu1.nm_program1",
        "left_Menu1.nm_program2",
        "left_Menu1.nm_program3",
    ]
    for mid in menu_ids:
        try:
            btn = wait.until(EC.element_to_be_clickable((By.ID, mid)))
            btn.click()
            time.sleep(0.5)
            handle_alert(driver)
        except Exception as e:
            print(f"⚠️ 메뉴 클릭 실패 [{mid}]: {e}")

    driver.switch_to.default_content()

    try:
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "haksa_hakjuk_v_hakjuk")))
    except TimeoutException:
        print("❌ 성적 프레임 진입 실패")
        return False

    return True


def capture_grade_response(driver, wait: WebDriverWait) -> Optional[str]:
    """
    이수학기별성적 클릭 후 selenium-wire에서 응답 본문 수집
    """
    print("3) 성적 데이터 요청...")
    driver.requests.clear()

    clicked = wait_and_click(
        driver,
        wait,
        By.XPATH,
        "//label[.//div[normalize-space(text())='이수학기별성적']]",
        desc="이수학기별성적"
    )
    if not clicked:
        return None

    target_req = None
    end_time = time.time() + 15

    while time.time() < end_time:
        for req in driver.requests:
            try:
                if req.method == "POST" and "crossurl.jsp" in req.url and req.response:
                    body = req.body or b""
                    if b"_my_TargetObject=TGlzdDM" in body:
                        target_req = req
                        break
            except Exception:
                continue
        if target_req:
            break
        time.sleep(0.2)

    if not target_req:
        candidates = []
        for r in driver.requests:
            try:
                if r.response and "crossurl.jsp" in r.url:
                    candidates.append(r)
            except Exception:
                pass

        if candidates:
            target_req = max(candidates, key=lambda r: len(r.response.body or b""))
            print("⚠️ 정확한 타겟 요청 탐지 실패, 가장 큰 응답으로 대체")
        else:
            print("❌ 성적 응답 요청을 찾지 못했습니다.")
            return None

    resp_bytes = target_req.response.body or b""

    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            resp_text = resp_bytes.decode(enc)
            print(f"✅ 응답 디코딩 성공: {enc}")
            return resp_text
        except Exception:
            continue

    resp_text = resp_bytes.decode("utf-8", errors="ignore")
    print("⚠️ 강제 utf-8(ignore) 디코딩 사용")
    return resp_text


# =========================
# 메인 함수
# =========================
def crawl_student_data(student_id: str, password: str, headless: bool = True) -> pd.DataFrame | None:
    """
    로그인 후 성적 데이터를 크롤링해서 과목 상세 DataFrame만 반환

    반환 기준:
    - 로그인 실패 / 페이지 진입 실패 -> None
    - 로그인 성공 + 데이터 없음 -> 빈 DataFrame
    """
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    if headless:
        chrome_options.add_argument("--headless=new")

    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--allow-running-insecure-content")

    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        wait = WebDriverWait(driver, 20)

        ok = login_and_navigate(driver, wait, student_id, password)
        if not ok:
            return None

        resp_text = capture_grade_response(driver, wait)
        if resp_text is None:
            return pd.DataFrame()

        df = parse_to_dataframe(resp_text)
        return df

    except UnexpectedAlertPresentException as e:
        print(f"🚨 예상치 못한 팝업 발생: {e.alert_text}")
        return None
    except WebDriverException as e:
        print(f"❌ WebDriver 오류: {e}")
        return None
    except Exception as e:
        print(f"❌ 크롤링 치명적 오류: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# =========================
# 학기별 보기용 헬퍼
# =========================
def split_by_semester(subject_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    과목 상세 DataFrame을 '2023-2' 같은 키로 분리
    """
    if subject_df.empty:
        return {}

    result = {}
    for key, group in subject_df.groupby("학기키", sort=False):
        result[key] = group.reset_index(drop=True)
    return result


def print_semester_report(subject_df: pd.DataFrame):
    """
    과목 DataFrame 기준 간단 출력용
    """
    if subject_df.empty:
        print("출력할 데이터가 없습니다.")
        return

    semester_keys = sorted(set(subject_df["학기키"].tolist()), reverse=True)

    for key in semester_keys:
        print("=" * 60)
        print(f"[{key}]")

        sem_subjects = subject_df[subject_df["학기키"] == key].copy()

        if not sem_subjects.empty:
            print(sem_subjects[["과목명", "학점", "성적", "이수구분", "교수명"]].to_string(index=False))
        else:
            print("과목 데이터 없음")