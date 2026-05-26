from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import socketio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware


from modules.agent_speed.route import router as agent_speed_router
from modules.auto_chet.route import router as auto_chet_router
from modules.chat.route import router as chat_router
from modules.chroma.route import router as chroma_router
from modules.log.route import router as log_router
from modules.login.route import router as login_router
from modules.rag.route import router as rag_router
from settings.config import SECRET_KEY, STATIC_DIR


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def split_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def get_allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if raw:
        return split_csv_env(raw)

    return [
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:20001",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:20001",
        "http://210.115.229.18",
        "http://210.115.229.18:20001",
    ]


APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "20002"))
APP_RELOAD = to_bool(os.getenv("RELOAD"), default=False)

SESSION_SECRET_KEY = (SECRET_KEY or "").strip() or os.getenv(
    "SECRET_KEY",
    "change-this-secret-key",
)
SESSION_HTTPS_ONLY = to_bool(os.getenv("SESSION_HTTPS_ONLY"), default=False)

STATIC_PATH = Path(STATIC_DIR)
ALLOWED_ORIGINS = get_allowed_origins()


def ok(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, **kwargs}


def safe_json_string(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def create_app() -> FastAPI:
    app = FastAPI(title="fastapi-rag-chat", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET_KEY,
        max_age=60 * 60 * 24,
        https_only=SESSION_HTTPS_ONLY,
        same_site="lax",
    )

    if STATIC_PATH.exists() and STATIC_PATH.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
        logger.info("Mounted /static -> %s", STATIC_PATH)
    else:
        logger.warning("STATIC_DIR not found: %s", STATIC_PATH)

    app.include_router(login_router)
    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(agent_speed_router)
    app.include_router(chroma_router)
    app.include_router(log_router)
    app.include_router(auto_chet_router)

    app.state.service_name = "fastapi-rag-chat"
    return app


app_fastapi = create_app()

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=ALLOWED_ORIGINS,
    logger=False,
    engineio_logger=False,
)
app_fastapi.state.sio = sio


@sio.event
async def connect(sid: str, environ: dict, auth: Any) -> None:
    logger.info("Socket connected: %s", sid)


@sio.event
async def disconnect(sid: str) -> None:
    logger.info("Socket disconnected: %s", sid)


@sio.event
async def ping(sid: str, data: Any) -> None:
    await sio.emit("pong", {"ok": True, "data": data}, to=sid)


@app_fastapi.get("/")
async def root() -> dict[str, Any]:
    return ok(
        service=app_fastapi.state.service_name,
        docs="/docs",
        health="/api/health",
        socketio="/socket.io/",
    )


@app_fastapi.get("/api/health")
async def api_health() -> dict[str, Any]:
    return ok(
        service=app_fastapi.state.service_name,
        socketio_enabled=True,
        static_enabled=STATIC_PATH.exists() and STATIC_PATH.is_dir(),
    )


@app_fastapi.get("/api/healthz")
async def api_healthz() -> dict[str, bool]:
    return {"ok": True}


@app_fastapi.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else safe_json_string(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": detail, "path": request.url.path},
    )


@app_fastapi.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "path": request.url.path},
    )


app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app_fastapi,
    socketio_path="socket.io",
)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_RELOAD,
    )
