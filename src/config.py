import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'app.db'}")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
RETRIEVAL_CACHE_DIR = BASE_DIR / "retrieval_cache"
