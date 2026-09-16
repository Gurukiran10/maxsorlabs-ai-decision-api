"""Builds (or rebuilds) the local RAG embedding cache from knowledge_base/*.md.

Usage:
    python -m scripts.ingest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import build_index, load_all_chunks  # noqa: E402


def main() -> None:
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks from knowledge_base/.")
    build_index(force=True)
    print("Embedding index built at retrieval_cache/.")


if __name__ == "__main__":
    main()
