"""Application configuration — always reads from os.environ (never cached)."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Paths ──
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
MODEL_DIR = DATA_DIR / "models"

for d in [UPLOAD_DIR, CHROMA_DIR, CHECKPOINT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Helpers — always read live from os.environ ──

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

# ── LLM Providers (live reads) ──
def get_anthropic_key(): return _env("ANTHROPIC_API_KEY")
def get_deepseek_key(): return _env("DEEPSEEK_API_KEY")
def get_openai_key(): return _env("OPENAI_API_KEY")
def get_voyage_key(): return _env("VOYAGE_API_KEY")
def get_default_provider(): return _env("DEFAULT_LLM_PROVIDER", "deepseek")
def get_password(): return _env("ACCESS_PASSWORD", "")

# ── Models ──
ANTHROPIC_DEFAULT_MODEL = os.environ.get("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-6")
DEEPSEEK_DEFAULT_MODEL = os.environ.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-v4-pro")
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4.1")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2")

# ── Embedding ──
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ── App ──
PORT = int(os.environ.get("PORT", "7860"))

# ── Limits ──
MAX_FILE_SIZE_MB = 50
MAX_IMAGE_PX = 1568
IMAGE_QUALITY = 85
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_TOP_K = 8
RERANK_TOP_K = 3
BGE_IDLE_UNLOAD_SEC = 1800
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

# ── Supported file types ──
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".txt", ".md", ".py", ".java", ".c", ".cpp",
    ".cs", ".js", ".ts", ".go", ".rs", ".html", ".css"
}

CODE_EXTENSIONS = {".py", ".java", ".c", ".cpp", ".cs", ".js", ".ts", ".go", ".rs"}
