"""清 — AI 学习助手 v2.0"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import PORT, DATA_DIR

# ── Logging ──
LOG_FILE = DATA_DIR / "qing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
logger = logging.getLogger("qing")

from app.routers import setup, upload, chat, actions

app = FastAPI(title="清 — AI Study Assistant", version="2.0.0")

# CORS — allow local access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# ── Main Page — / ──
@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


# ── Health Check ──
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "app": "清",
        "version": "2.0.0",
    }


# ── API Routers ──
app.include_router(setup.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(actions.router)

# ── Static Files (CSS, JS, etc.) — /static/xxx ──
app.mount("/static", StaticFiles(directory=static_dir.as_posix()), name="static")

# ── Startup ──
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  清 — AI Study Assistant")
    print(f"  Starting at http://localhost:{PORT}")
    print("=" * 50)
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
