import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INFO_DIR = os.path.join(BASE_DIR, "info")

SECRET_KEY = os.getenv("SECRET_KEY", "default_fastapi_fallback_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

ANALYSIS_TTL_HOURS = int(os.getenv("ANALYSIS_TTL_HOURS", "12"))

os.makedirs(INFO_DIR, exist_ok=True)