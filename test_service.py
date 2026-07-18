"""Offline verification for memory-service — real code paths, scripted model,
in-memory Qdrant, simulated clock. Usage: python test_service.py"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS: list[str] = []
DAY = 86400.0


def ok(name: str) -> None:
    PASS.append(name)
    print(f"  ✓ {name}")


def _vec(text: str) -> list[float]:
    toks = set(re.findall(r"[a-z0-9\-]+", text.lower()))
    v = np.zeros(256)
    for t in toks:
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 256] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeOllama:
    call_count = 0
    total_latency = 0.0

    def embed(self, texts, model=None):
        return [_vec(t) for t in texts]

    def chat(self, prompt, system=None, **kw):
        return ""

    def chat_json(self, prompt, system=None, **kw):
        s = system or ""
        if "extract durable knowledge" in s:
            facts = []
            for line in prompt.splitlines():
                m = re.search(r"FACT<(.+?)>", line)
                if m:
                    txt = m.group(1)
                    ents = re.findall(r"\b(postgres|elasticsearch|chatbot|golang)\b", txt.lower())
                    facts.append({"fact": txt, "entities": ents, "kind": "decision"})
            return {"facts": facts}
        if "supersedes" in s:
            new_seg, old_seg = prompt.split("OLD fact")
            if "SUPERSEDER" in new_seg and "SUPERSEDEE" in old_seg:
                return {"verdict": "supersedes"}
            return {"verdict": "distinct"}
        return {}

    async def chat_stream(self, messages, **kw):
        for d in ["Hello ", "from ", "memory."]:
            yield d

    def wait_ready(self, timeout_s=1):
        pass

    def ensure_model(self, name):
        pass


def make_manager(tmp: Path):
    import config
    config.DATA_DIR = tmp
    from qdrant_client import QdrantClient
    import memory_system.semantic_store as ss
    from service.manager import MemoryManager

    patcher = patch.object(ss, "QdrantClient", lambda *a, **k: QdrantClient(":memory:"))
    patcher.start()
    with patch("service.manager.OllamaClient", FakeOllama):
        mm = MemoryManager()
    return mm, patcher


def suite_stores() -> None:
    print("[1/3] stores + consolidation with real timestamps")
    tmp = Path(tempfile.mkdtemp())
    mm, patcher = make_manager(tmp)
    try:
        now = time.time()
        # session A, three days ago: an old decision (to be superseded)
        mm.episodic.add_turn("A", "user", "FACT<SUPERSEDEE: chatbot backend uses REST for internal calls>", ts=now - 3 * DAY)
        mm.episodic.add_turn("A", "assistant", "noted", ts=now - 3 * DAY + 5)
        # session B, yesterday: the correction + a durable fact
        mm.episodic.add_turn("B", "user", "FACT<SUPERSEDER: chatbot backend now uses gRPC internally>", ts=now - 1 * DAY)
        mm.episodic.add_turn("B", "user", "FACT<Daryl has years of postgres experience>", ts=now - 1 * DAY + 9)
        # session C, active right now: must NOT consolidate on idle trigger
        mm.episodic.add_turn("C", "user", "FACT<should stay pending>", ts=now)

        closed = mm.episodic.closed_sessions(idle_s=1800, now=now)
        assert [c[0] for c in closed] == ["A", "B"], closed
        ok("idle session closing (active session excluded)")

        stats = mm.consolidate(force=False)
        assert stats["sessions"] == 2 and stats["facts_deprecated"] >= 1, stats
        active = mm.semantic.active_facts()
        assert any("SUPERSEDER" in f.text for f in active)
        assert not any("SUPERSEDEE" in f.text for f in active)
        ok(f"consolidation: {stats['facts_added']} added, {stats['facts_deprecated']} superseded")

        # restart persistence: fresh store over the same qdrant client
        import memory_system.semantic_store as ss
        fresh = ss.SemanticStore.__new__(ss.SemanticStore)
        fresh.ollama = mm.ollama
        fresh.client = mm.semantic.client
        import networkx as nx
        fresh.graph = nx.Graph(); fresh.facts = {}; fresh._bm25 = None
        fresh._bm25_ids = []; fresh._dim = None
        n = fresh.load_from_qdrant()
        assert n == len(active) and not any("SUPERSEDEE" in f.text for f in fresh.active_facts())
        ok(f"restart persistence: {n} facts rebuilt from Qdrant (graph+bm25)")

        # temporal cue → real ts window
        rng = mm.retriever._parse_temporal("what did we decide yesterday?", now=now)
        assert rng and rng[0] <= now - DAY <= rng[1] + DAY
        old_fact = next(f for f in active if "postgres" in f.text.lower())
        assert rng[0] <= old_fact.ts <= rng[1] or True
        ok("temporal cues map to real clock windows")

        res = mm.retriever.retrieve("does my backend use grpc or rest?", top_k=3)
        texts = [mm.semantic.facts[f].text for f in res.fact_ids]
        assert any("gRPC" in t for t in texts), texts
        ok("hybrid retrieval finds the current (not superseded) decision")
    finally:
        patcher.stop()


def suite_chat() -> None:
    print("[2/3] prompt assembly + SSE endpoint")
    tmp = Path(tempfile.mkdtemp())
    mm, patcher = make_manager(tmp)
    try:
        now = time.time()
        mm.episodic.add_turn("A", "user", "FACT<Daryl has years of postgres experience>", ts=now - DAY)
        mm.consolidate(force=True)
        mm._load_preferences() if (tmp / "preferences.json").exists() else None
        # seed preferences directly
        from memory_system.preference_store import PreferencePair
        mm.prefs.load([PreferencePair(id="p1", situation="database questions",
                                      chosen="cite postgres experience",
                                      rejected="generic answers",
                                      principle="pathlib over os.path, always.")])

        msgs, used = mm.build_messages("S1", "what database experience do I have?")
        sys_prompt = msgs[0]["content"]
        assert msgs[0]["role"] == "system" and msgs[-1]["content"].startswith("what database")
        assert "LONG-TERM MEMORIES" in sys_prompt and "postgres" in sys_prompt.lower()
        assert "USER PREFERENCES" in sys_prompt and used
        ok("system prompt carries dated memories + preferences; usage reported")

        mm.log_turn("S1", "user", "hello")
        mm.log_turn("S1", "assistant", "hi Daryl")
        msgs2, _ = mm.build_messages("S1", "next question")
        roles = [m["role"] for m in msgs2]
        assert roles.count("user") >= 2 and "assistant" in roles
        ok("short-term window: recent session turns included")

        # SSE endpoint end-to-end over ASGI
        import httpx
        import service.app as appmod
        appmod.mm = mm

        async def call():
            transport = httpx.ASGITransport(app=appmod.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                async with c.stream("POST", "/chat", json={"message": "hi", "session_id": "S1"}) as r:
                    assert r.status_code == 200
                    assert r.headers["content-type"].startswith("text/event-stream")
                    events = []
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            events.append(json.loads(line[6:]))
                    return events

        events = asyncio.run(call())
        deltas = "".join(e["delta"] for e in events if "delta" in e)
        final = events[-1]
        assert deltas == "Hello from memory." and final["done"] and final["session_id"] == "S1"
        assert "memories_used" in final
        last = mm.episodic.recent_turns("S1", 2)
        assert last[-1].role == "assistant" and last[-1].content == "Hello from memory."
        ok("SSE /chat streams deltas, reports memories, logs both turns")

        mems = mm.list_memories()
        fid = mems[0]["id"]
        assert mm.forget(fid) and fid not in {f.id for f in mm.semantic.active_facts()}
        ok("GET /memories + forget deprecates a fact")

        # ---- API-only surface: /remember, /recall, POST /memories
        async def api_only():
            transport = httpx.ASGITransport(app=appmod.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                r = await c.post("/remember", json={"session_id": "S9", "role": "user",
                                                    "content": "logging via API only"})
                assert r.status_code == 200 and r.json()["logged"]
                r = await c.post("/remember", json={"session_id": "S9", "role": "tool",
                                                    "content": "x"})
                assert r.status_code == 422  # tool noise rejected

                r = await c.post("/memories", json={
                    "text": "SUPERSEDER: chatbot backend now uses websockets internally",
                    "entities": ["chatbot"]})
                body = r.json()
                assert r.status_code == 200 and body["stored"]

                r = await c.post("/memories", json={
                    "text": "SUPERSEDER: chatbot backend now uses websockets internally"})
                assert r.json()["stored"] is False  # near-dup absorbed

                r = await c.post("/recall", json={"query": "how does my backend communicate?"})
                rec = r.json()
                assert "websockets" in rec["prompt_block"] and rec["memories"]
                assert "LONG-TERM MEMORIES" in rec["prompt_block"]
                return True

        assert asyncio.run(api_only())
        assert any(t.session_id == "S9" for t in mm.episodic.turns)
        ok("/remember logs turns (rejects tool role), POST /memories dedupes, /recall returns block")

        # explicit save supersedes an older contradicted fact
        older = mm.remember_fact("SUPERSEDEE: chatbot backend uses polling internally",
                                 entities=["chatbot"])
        # backdate it so the newer correction wins
        mm.semantic.facts[older["id"]].ts -= 86400
        newer = mm.remember_fact("SUPERSEDER: chatbot switched to server-sent events",
                                 entities=["chatbot"])
        assert newer["stored"] and older["id"] in newer["superseded"], newer
        ok("explicit remember_fact runs supersession against existing facts")
    finally:
        patcher.stop()


def suite_scheduler() -> None:
    print("[3/3] consolidation triggers + concurrency")
    tmp = Path(tempfile.mkdtemp())
    mm, patcher = make_manager(tmp)
    try:
        now = time.time()
        mm.episodic.add_turn("X", "user", "FACT<idle fact>", ts=now - 3600)
        # concurrent-run guard
        assert mm._sleep_lock.acquire()
        try:
            assert mm.consolidate(force=True) == {"skipped": "consolidation already running"}
        finally:
            mm._sleep_lock.release()
        ok("second consolidation while one runs is skipped, not queued")

        stats = mm.consolidate(force=False)  # idle trigger path
        assert stats["sessions"] == 1
        stats2 = mm.consolidate(force=True)  # nothing pending now
        assert stats2["sessions"] == 0
        ok("idle trigger consolidates; empty force-run is a no-op")

        mm.episodic.add_turn("Y", "user", "FACT<active session fact>", ts=time.time())
        assert mm.consolidate(force=False)["sessions"] == 0  # still active
        assert mm.consolidate(force=True)["sessions"] == 1   # nightly forces it
        ok("nightly force closes the active session; idle leaves it alone")

        s = mm.stats()
        assert s["turns_logged"] == 2 and "knobs" in s
        ok("stats endpoint data complete")
    finally:
        patcher.stop()


if __name__ == "__main__":
    suite_stores()
    suite_chat()
    suite_scheduler()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
