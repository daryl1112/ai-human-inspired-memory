"""Hybrid multi-cue retrieval (benchmark 3 engine).

Three signals — dense vectors, BM25 lexical, knowledge-graph association —
fused with reciprocal rank fusion, plus a temporal-cue parser that turns
human vagueness ('early on', 'last week') into a simulated-day filter.
Optional LLM rerank of the fused pool.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from config import Knobs
from memory_system.ollama_client import OllamaClient
from memory_system.semantic_store import SemanticStore

_DAY = 86400.0
# pattern -> (days_back_start, days_back_end) relative to now
_TEMPORAL_CUES: list[tuple[str, tuple[float, float]]] = [
    (r"just now|a moment ago|today|this morning", (1.0, 0.0)),
    (r"yesterday|the other day|last night", (3.0, 0.0)),
    (r"recently|lately|these days|latest", (10.0, 0.0)),
    (r"last week|past week|a week ago|about a week", (14.0, 4.0)),
    (r"last month|a month ago|weeks ago", (45.0, 14.0)),
    (r"a while back|way back|originally|early on|at the (very )?start|beginning", (3650.0, 30.0)),
]

_RERANK_SYS = (
    "Rank the candidate memory facts by relevance to the query. Return strict "
    'JSON: {"ranking": [fact numbers, best first]}. Consider that the query '
    "may be vague and refer to the fact obliquely."
)


@dataclass
class RetrievalResult:
    fact_ids: list[str]
    latency_s: float
    ts_range: tuple[float, float] | None


class HybridRetriever:
    def __init__(self, ollama: OllamaClient, semantic: SemanticStore, knobs: Knobs):
        self.ollama = ollama
        self.semantic = semantic
        self.knobs = knobs

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        t0 = time.time()
        ts_range = self._parse_temporal(query)

        dense = self.semantic.dense_search(query, self.knobs.top_k_dense, ts_range)
        bm25 = self.semantic.bm25_search(query, self.knobs.top_k_bm25)
        graph = self.semantic.graph_search(query, self.knobs.graph_hops)

        fused = self._rrf(
            [
                [fid for fid, _ in dense],
                [fid for fid, _ in bm25],
                graph,
            ]
        )
        if ts_range:  # soft temporal boost for non-dense signals
            fused = self._temporal_filter(fused, ts_range)

        if self.knobs.rerank and len(fused) > top_k:
            fused = self._rerank(query, fused[: self.knobs.rerank_pool]) + fused[
                self.knobs.rerank_pool :
            ]
        return RetrievalResult(
            fact_ids=fused[:top_k],
            latency_s=time.time() - t0,
            ts_range=ts_range,
        )

    # ------------------------------------------------------------------ util
    @staticmethod
    def _parse_temporal(query: str, now: float | None = None
                        ) -> tuple[float, float] | None:
        q = query.lower()
        now = now if now is not None else time.time()
        for pattern, (back_start, back_end) in _TEMPORAL_CUES:
            if re.search(pattern, q):
                return (now - back_start * _DAY, now - back_end * _DAY)
        return None

    def _rrf(self, rankings: list[list[str]]) -> list[str]:
        k = self.knobs.rrf_k
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, fid in enumerate(ranking):
                scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank + 1)
        return [fid for fid, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def _temporal_filter(self, fids: list[str], rng: tuple[float, float]) -> list[str]:
        inside, outside = [], []
        for fid in fids:
            f = self.semantic.facts.get(fid)
            (inside if f and rng[0] <= f.ts <= rng[1] else outside).append(fid)
        return inside + outside

    def _rerank(self, query: str, fids: list[str]) -> list[str]:
        listing = "\n".join(
            f"{i + 1}. {self.semantic.facts[fid].text}" for i, fid in enumerate(fids)
        )
        try:
            out = self.ollama.chat_json(
                f"QUERY: {query}\n\nCANDIDATES:\n{listing}",
                system=_RERANK_SYS,
                num_ctx=4096,
                temperature=0.0,
                num_predict=512,
            )
            order = out.get("ranking", []) if isinstance(out, dict) else []
            seen, result = set(), []
            for n in order:
                idx = int(n) - 1
                if 0 <= idx < len(fids) and idx not in seen:
                    seen.add(idx)
                    result.append(fids[idx])
            result += [fid for i, fid in enumerate(fids) if i not in seen]
            return result
        except (ValueError, TypeError):
            return fids
