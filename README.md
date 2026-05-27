# Academic Agent

Academic Agent는 대학/기관 웹페이지, 공지사항, 학사 문서, 교육과정표, 졸업요건, 강의계획서 등의 데이터를 기반으로 질의응답을 수행하는 **FastAPI 기반 RAG 챗봇 시스템**입니다.

예시 도메인은 **한림대학교 학사/교육과정/공지/졸업요건 데이터**를 기준으로 구성되어 있습니다.

본 프로젝트는 웹페이지와 문서에서 수집한 비정형 데이터를 검색 가능한 형태로 전처리하고, ChromaDB 기반 벡터 검색과 LLM 응답 생성을 결합하여 근거 기반 답변을 제공하는 것을 목표로 합니다.

> 본 저장소는 Academic Agent의 메인 애플리케이션 구조를 공개합니다.  
> 일부 핵심 검색/전처리 레이어는 별도 비공개 서비스로 운영됩니다.

---

## Demo

### 체험 URL

[http://bis.hallym.ac.kr/](http://bis.hallym.ac.kr/)

### 시연 영상

[예제 영상 보기](https://bis.hallym.ac.kr/static/video/example.mp4)

### 테스트 계정

```text
ID: user_bis
PW: password_bis
```

> 위 계정은 서비스 체험을 위한 테스트 계정입니다.  
> 운영 계정이나 실제 사용자 계정이 아닙니다.  
> 실제 성적 정보는 학사정보시스템에 로그인 가능한 계정에서만 확인할 수 있습니다.

---

## Overview

Academic Agent는 사용자 질문을 분석한 뒤, 내부 문서 검색 또는 외부 웹 검색이 필요한지 판단하고 RAG 기반 답변을 생성합니다.

```text
사용자 질문
   ↓
Auto Router / Agent Planner
   ↓
Internal Search 또는 External Search
   ↓
문서 검색 / 웹페이지 수집 / 본문 추출
   ↓
RAG Context 구성
   ↓
Ollama 또는 OpenAI 기반 답변 생성
   ↓
Web UI / SSE Streaming / Kakao Callback 응답
```

처리 가능한 질문 예시는 다음과 같습니다.

```text
한림대학교 졸업요건 알려줘
한림대학교 학사일정 중 5월 주요 일정 알려줘
국어국문학전공 1학년 교과과정 알려줘
AI의료융합전공 나노디그리 교과목 알려줘
미디어커뮤니케이션전공 졸업요건 요약해줘
```

---

## Key Features

- 학사 문서 및 웹페이지 기반 RAG 질의응답
- FastAPI 기반 백엔드
- ChromaDB 기반 벡터 검색
- 문서 업로드 및 벡터화
- URL 기반 웹페이지 수집
- HTML 본문 및 테이블 전처리 연동
- Ollama 로컬 LLM 및 OpenAI API 연동
- SSE 기반 실시간 스트리밍 응답
- 카카오 챗봇 Callback 응답 지원
- 관리자 로그인 및 로그 저장
- 학사/진로/교과목 추천 응답 기능

---

## Architecture

```text
User / Web UI / Kakao
        ↓
FastAPI Backend
        ↓
Auto Router
        ↓
Agent / RAG Pipeline
        ↓
Internal Search / External Search
        ↓
ChromaDB / Web Fetch / Content Extraction
        ↓
Ollama or OpenAI
        ↓
Streaming / Callback Response
```

Academic Agent는 크게 다음 구성으로 동작합니다.

```text
- FastAPI Application
- Auto Router
- Agent Layer
- Internal Search Layer
- External Search Layer
- ChromaDB Vector Store
- LLM Response Generator
- Streaming / Callback Layer
```

---

## Public / Private Scope

본 프로젝트는 전체 Academic Agent 시스템 중 메인 애플리케이션 구조와 RAG 처리 흐름을 공개합니다.

### 공개 범위

```text
- FastAPI 서버 구조
- Agent 서비스 구조
- ChromaDB 연동 구조
- 문서 업로드/검색 구조
- URL 기반 웹페이지 수집 구조
- HTML 본문 추출 연동 구조
- 로그/세션/관리자 UI 구조
- RAG 처리 흐름
```

### 비공개 범위

```text
- Google Search Layer 전체 구현
- HTML Table Preprocessing Layer 전체 구현
```

> Google Search Layer와 HTML Table Preprocessing Layer는 시스템에서 사용되지만, 본 저장소에는 전체 구현을 공개하지 않습니다.

---

## Main Use Cases

### 1. 학사 정보 질의응답

```text
한림대학교 졸업요건 알려줘
한림대학교 학사일정 알려줘
수강신청 일정 알려줘
```

### 2. 교육과정 검색

```text
국어국문학전공 1학년 교과과정 알려줘
AI의료융합전공 나노디그리 교과목 알려줘
미디어커뮤니케이션전공 졸업요건 요약해줘
```

### 3. 문서 기반 RAG

```text
업로드한 PDF 기준으로 졸업요건 설명해줘
공지사항 문서에서 장학금 관련 내용 찾아줘
학과별 교육과정표에서 2학년 과목 알려줘
```

### 4. 진로/교과목 추천

```text
AI 의료 분야로 가고 싶은데 어떤 과목을 들으면 좋을까?
백엔드 개발자가 되고 싶은데 어떤 수업을 들으면 좋을까?
데이터 분석 쪽 진로를 생각하면 어떤 전공 과목이 좋아?
```

---

## Supported Interfaces

```text
- Web UI
- RAG Chat
- Agent Search
- Document Upload
- URL Ingestion
- Student Data Interface
- Course Recommendation
- Kakao Callback
- Admin Log View
```

---

## Supported File Types

문서 업로드 및 벡터화 기능은 다음 파일 형식을 지원합니다.

```text
.txt
.md
.pdf
.docx
.pptx
.hwpx
.hwp
```

---

## Technology Stack

### Backend

```text
Python
FastAPI
Uvicorn
Socket.IO
SSE Streaming
Starlette SessionMiddleware
```

### LLM / Agent

```text
Ollama
OpenAI API
LangChain
LangGraph
```

### Vector DB / RAG

```text
ChromaDB
Semantic Chunking
Reranking
HTML Content Extraction
```

### Parsing / Preprocessing

```text
BeautifulSoup
lxml
html5lib
PyMuPDF
pymupdf4llm
python-docx
python-pptx
```

### Private Layers

```text
Google Search Layer
HTML Table Preprocessing Layer
```

---

## Installation

```bash
git clone <repository-url>
cd academic-agent-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

프로젝트 루트에 `.env` 파일을 생성하고 실행 환경에 맞게 설정합니다.

```env
OPENAI_API_KEY=
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=
HTML_TABLE_API_BASE_URL=
GOOGLE_SEARCH_LAYER_URL=
SECRET_KEY=
```

> 운영용 secret, token, private endpoint, 실제 사용자 데이터는 공개 저장소에 포함하지 않습니다.

---

## Run

### 1. Start Ollama

```bash
ollama serve
```

필요한 모델을 내려받습니다.

```bash
ollama pull <model-name>
```

### 2. Start FastAPI Server

```bash
python main.py
```

또는:

```bash
uvicorn main:app --host 0.0.0.0 --port 20002
```

실행 후 브라우저에서 확인합니다.

```text
http://localhost:20002
```

---

## Usage

### Demo Login

```text
URL: http://bis.hallym.ac.kr/
ID: user_bis
PW: password_bis
```

### Example Questions

```text
한림대학교 졸업요건 알려줘
한림대학교 학사일정 알려줘
국어국문학전공 1학년 교과과정 알려줘
AI의료융합전공 나노디그리 교과목 알려줘
AI 의료 분야로 가고 싶은데 어떤 과목을 들으면 좋을까?
```

---

## Roadmap

```text
- HTML Table Preprocessing Layer 안정화
- table-aware chunking 추가
- Google Search Layer와 Agent Planner 간 score 통합
- collection별 embedding model 설정
- benchmark 데이터셋 작성
```

---

## License

## Custom Restricted License

본 프로젝트는 작성자의 명시적인 사전 허가 없이 사용할 수 없는 제한 라이선스를 따릅니다.

본 저장소의 코드, 문서, 구조, 아이디어, 예제, 전처리 방식, 검색 레이어 설계, 테이블 정규화 방식 및 파생물은 작성자의 사전 서면 허가 없이 사용할 수 없습니다.
