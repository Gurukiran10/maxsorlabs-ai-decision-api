"""Local RAG pipeline: chunk policy docs, embed with Gemini, retrieve by cosine similarity.

No vector DB is used. Embeddings for the (small, static) knowledge base are
computed once and cached to disk as a .npz file; retrieval is a plain NumPy
cosine-similarity scan, which is more than sufficient for ~30 short chunks.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google import genai

from src.config import GEMINI_API_KEY, KNOWLEDGE_BASE_DIR, RETRIEVAL_CACHE_DIR

EMBEDDING_MODEL = "text-embedding-004"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


@dataclass
class Chunk:
    doc: str
    text: str


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a policy doc into one chunk per numbered rule (plus the title).

    These docs are short numbered-list policies, so a rule-per-chunk split
    keeps each chunk atomic and semantically self-contained, which gives much
    more precise retrieval than fixed-size windows would for text this short.
    """
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else path.stem
    chunks = [Chunk(doc=path.name, text=f"{title} (overview): {path.stem.replace('_', ' ')} policy document.")]

    rule_pattern = re.compile(r"^\d+\.\s*")
    for line in lines[1:]:
        if rule_pattern.match(line):
            rule_text = rule_pattern.sub("", line)
            chunks.append(Chunk(doc=path.name, text=f"{title}: {rule_text}"))
    return chunks


def load_all_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    client = _get_client()
    vectors = []
    # Embed one at a time for compatibility across SDK versions / batch limits.
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"task_type": task_type},
        )
        vectors.append(result.embeddings[0].values)
    return np.array(vectors, dtype=np.float32)


def build_index(force: bool = False) -> None:
    RETRIEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path = RETRIEVAL_CACHE_DIR / "vectors.npy"
    meta_path = RETRIEVAL_CACHE_DIR / "chunks.json"

    if vectors_path.exists() and meta_path.exists() and not force:
        return

    chunks = load_all_chunks()
    embeddings = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    np.save(vectors_path, embeddings)
    meta_path.write_text(
        json.dumps([{"doc": c.doc, "text": c.text} for c in chunks], indent=2), encoding="utf-8"
    )


def _load_index() -> tuple[np.ndarray, list[Chunk]]:
    build_index(force=False)
    vectors = np.load(RETRIEVAL_CACHE_DIR / "vectors.npy")
    meta = json.loads((RETRIEVAL_CACHE_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(doc=m["doc"], text=m["text"]) for m in meta]
    return vectors, chunks


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm


def retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    vectors, chunks = _load_index()
    query_vec = embed_texts([query_text], task_type="RETRIEVAL_QUERY")[0]
    scores = cosine_similarity(query_vec, vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"doc": chunks[i].doc, "text": chunks[i].text, "score": float(scores[i])}
        for i in top_indices
    ]
