"""Semantic store: consolidated facts held in three co-indexed structures.

  * Qdrant collection  -> dense embeddings (payload carries fact metadata)
  * rank_bm25 index    -> lexical signal, rebuilt after each consolidation
  * NetworkX graph     -> entity nodes <-> fact nodes for associative hops

A fact is 'active' or 'deprecated'. Deprecated facts stay in Qdrant for
audit but are filtered out of every retrieval path — that filter IS the
forgetting mechanism.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

import config
from memory_system.ollama_client import OllamaClient

COLLECTION = "chat_memory_facts"


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-]+", text.lower())


@dataclass
class Fact:
    id: str
    text: str
    entities: list[str]
    ts: float                   # unix time the fact was established
    status: str = "active"      # active | deprecated
    source_episodes: list[str] = field(default_factory=list)


class SemanticStore:
    def __init__(self, ollama: OllamaClient, qdrant_url: str = config.QDRANT_URL):
        self.ollama = ollama
        self.client = QdrantClient(url=qdrant_url, timeout=60)
        self.graph = nx.Graph()
        self.facts: dict[str, Fact] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._dim: int | None = None

    # ---------------------------------------------------------------- setup
    def reset(self) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.graph = nx.Graph()
        self.facts = {}
        self._bm25 = None
        self._bm25_ids = []

    def _ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._dim = dim

    # ----------------------------------------------------------------- write
    def add_facts(self, facts: list[Fact]) -> None:
        if not facts:
            return
        vecs = self.ollama.embed([f.text for f in facts])
        self._ensure_collection(len(vecs[0]))
        points = []
        for f, v in zip(facts, vecs):
            self.facts[f.id] = f
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f.id)),
                    vector=v,
                    payload={
                        "fact_id": f.id,
                        "text": f.text,
                        "entities": f.entities,
                        "ts": f.ts,
                        "status": f.status,
                    },
                )
            )
            self.graph.add_node(f.id, kind="fact")
            for ent in f.entities:
                e = ent.lower().strip()
                self.graph.add_node(e, kind="entity")
                self.graph.add_edge(f.id, e)
        self.client.upsert(COLLECTION, points=points)
        self._rebuild_bm25()

    def deprecate(self, fact_id: str) -> None:
        f = self.facts.get(fact_id)
        if not f or f.status == "deprecated":
            return
        f.status = "deprecated"
        if self.client.collection_exists(COLLECTION):
            self.client.set_payload(
                COLLECTION,
                payload={"status": "deprecated"},
                points=[str(uuid.uuid5(uuid.NAMESPACE_URL, fact_id))],
            )
        if self.graph.has_node(fact_id):
            self.graph.remove_node(fact_id)  # drop associative access too
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        active = self.active_facts()
        self._bm25_ids = [f.id for f in active]
        corpus = [_tok(f.text) for f in active]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    # ------------------------------------------------------------------ read
    def active_facts(self) -> list[Fact]:
        return [f for f in self.facts.values() if f.status == "active"]

    def active_bytes(self) -> int:
        return sum(len(f.text.encode()) for f in self.active_facts())

    def load_from_qdrant(self) -> int:
        """Rebuild in-memory indexes (facts dict, graph, BM25) from the
        persisted Qdrant collection after a service restart."""
        if not self.client.collection_exists(COLLECTION):
            return 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                COLLECTION, limit=256, offset=offset, with_payload=True
            )
            for p in points:
                pl = p.payload
                f = Fact(id=pl["fact_id"], text=pl["text"],
                         entities=pl.get("entities", []), ts=pl["ts"],
                         status=pl.get("status", "active"))
                self.facts[f.id] = f
                if f.status == "active":
                    self.graph.add_node(f.id, kind="fact")
                    for ent in f.entities:
                        e = ent.lower().strip()
                        self.graph.add_node(e, kind="entity")
                        self.graph.add_edge(f.id, e)
            if offset is None:
                break
        self._rebuild_bm25()
        return len(self.active_facts())

    def dense_search(
        self, query: str, top_k: int, ts_range: tuple[float, float] | None = None
    ) -> list[tuple[str, float]]:
        if not self.facts or not self.client.collection_exists(COLLECTION):
            return []  # empty store (e.g. consolidation failed) is not an error
        vec = self.ollama.embed([query])[0]
        must = [FieldCondition(key="status", match=MatchValue(value="active"))]
        flt = Filter(must=must)
        res = self.client.query_points(
            COLLECTION, query=vec, limit=top_k * 3 if ts_range else top_k,
            query_filter=flt, with_payload=True,
        ).points
        hits = []
        for p in res:
            d = p.payload["ts"]
            if ts_range and not (ts_range[0] <= d <= ts_range[1]):
                continue
            hits.append((p.payload["fact_id"], float(p.score)))
        return hits[:top_k]

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
        return [(fid, float(s)) for fid, s in ranked[:top_k] if s > 0]

    def graph_search(self, query: str, hops: int) -> list[str]:
        """Entity-anchored associative retrieval: match query tokens to entity
        nodes, walk `hops` and collect fact nodes."""
        qtok = set(_tok(query))
        seeds = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("kind") == "entity"
            and (n in qtok or any(t in n or n in t for t in qtok if len(t) > 3))
        ]
        found: set[str] = set()
        frontier = set(seeds)
        for _ in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                for nb in self.graph.neighbors(node):
                    if self.graph.nodes[nb].get("kind") == "fact":
                        found.add(nb)
                    else:
                        nxt.add(nb)
                    nxt.add(nb)
            frontier = nxt
        return list(found)
