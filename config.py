"""Service configuration. Retrieval knobs come from the benchmark's tuned
best_knobs.json when present (drop it into ./data), else safe defaults."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

WORKER_MODEL = os.environ.get("WORKER_MODEL", "qwen3:30b-a3b")      # answers chat
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", WORKER_MODEL)        # sleep-phase worker
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

SEED = 1151

# Consolidation triggers
IDLE_CONSOLIDATE_S = int(os.environ.get("IDLE_CONSOLIDATE_S", "1800"))   # 30 min idle
NIGHTLY_HOUR = int(os.environ.get("NIGHTLY_HOUR", "3"))                  # 03:00 local

# Chat behavior
SHORT_TERM_TURNS = int(os.environ.get("SHORT_TERM_TURNS", "12"))  # recent turns in-context
MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))
CHAT_NUM_CTX = int(os.environ.get("CHAT_NUM_CTX", "8192"))
CHAT_NUM_PREDICT = int(os.environ.get("CHAT_NUM_PREDICT", "1500"))

BASE_SYSTEM_PROMPT = os.environ.get(
    "BASE_SYSTEM_PROMPT",
    "You are Daryl's personal engineering assistant. Be direct, technical, "
    "and concise. Use the provided long-term memories when relevant; do not "
    "invent memories that are not listed.",
)


@dataclass
class Knobs:
    """Retrieval/consolidation knobs (schema-compatible with the benchmark)."""

    chunk_chars: int = 8000
    scratchpad_budget_tokens: int = 600
    cluster_sim_threshold: float = 0.72
    contradiction_sim_threshold: float = 0.80
    top_k_dense: int = 20
    top_k_bm25: int = 10
    graph_hops: int = 1
    rrf_k: int = 60
    rerank: bool = False
    rerank_pool: int = 12
    n_preference_exemplars: int = 3

    @classmethod
    def load(cls) -> "Knobs":
        p = DATA_DIR / "best_knobs.json"
        if p.exists():
            d = json.loads(p.read_text())
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()
