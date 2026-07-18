"""MemoryManager: the one object that owns all stores and implements the
three verbs of the service — remember (log turns), recall (retrieve + build
the augmented prompt), and sleep (consolidate closed sessions)."""
from __future__ import annotations

import json
import logging
import threading
import time

import config
from memory_system.consolidation import SleepPhase
from memory_system.episodic_store import EpisodicStore
from memory_system.ollama_client import OllamaClient
from memory_system.preference_store import PreferencePair, PreferenceStore
from memory_system.retrieval import HybridRetriever
from memory_system.semantic_store import SemanticStore

log = logging.getLogger("memory")


class MemoryManager:
    def __init__(self) -> None:
        self.knobs = config.Knobs.load()
        self.ollama = OllamaClient()
        self.episodic = EpisodicStore(config.DATA_DIR / "episodic.jsonl")
        self.semantic = SemanticStore(self.ollama)
        self.retriever = HybridRetriever(self.ollama, self.semantic, self.knobs)
        self.sleep = SleepPhase(self.ollama, self.episodic, self.semantic, self.knobs)
        self.prefs = PreferenceStore(self.ollama, self.knobs)
        self._sleep_lock = threading.Lock()
        self.last_sleep: dict = {}

    # ----------------------------------------------------------------- setup
    def start(self) -> None:
        self.ollama.wait_ready()
        self.ollama.ensure_model(config.WORKER_MODEL)
        self.ollama.ensure_model(config.EMBED_MODEL)
        restored = self.semantic.load_from_qdrant()
        log.info("restored %d active facts from Qdrant", restored)
        self._load_preferences()

    def _load_preferences(self) -> None:
        p = config.DATA_DIR / "preferences.json"
        if p.exists():
            pairs = [PreferencePair(**d) for d in json.loads(p.read_text())]
            if pairs:
                self.prefs.load(pairs)
                log.info("loaded %d preference pairs", len(pairs))

    # ------------------------------------------------------------------ chat
    def recall(self, query: str, top_k: int | None = None) -> dict:
        """Retrieve relevant memories + preferences for a query and return
        both a ready-to-inject prompt block and the structured facts. This is
        the API-only integration surface: callers paste `prompt_block` into
        their own system content."""
        used: list[dict] = []
        mem_lines: list[str] = []
        if self.semantic.active_facts():
            res = self.retriever.retrieve(query, top_k=top_k or config.MEMORY_TOP_K)
            for fid in res.fact_ids:
                f = self.semantic.facts.get(fid)
                if f:
                    when = time.strftime("%Y-%m-%d", time.localtime(f.ts))
                    mem_lines.append(f"- ({when}) {f.text}")
                    used.append({"id": f.id, "text": f.text, "ts": f.ts})

        parts: list[str] = []
        if mem_lines:
            parts.append("LONG-TERM MEMORIES (retrieved, dated):\n" + "\n".join(mem_lines))
        if self.prefs.pairs:
            rel = self.prefs.relevant(query)
            if rel:
                parts.append("USER PREFERENCES (follow these):\n" + "\n".join(
                    f"- {p.principle}" for p in rel
                ))
        return {"prompt_block": "\n\n".join(parts), "memories": used}

    def remember_fact(self, text: str, entities: list[str] | None = None) -> dict:
        """Explicit, immediate save (agent tool / user command) that bypasses
        the sleep phase but still gets dedup + supersession against existing
        facts, so a corrected fact deprecates what it replaces."""
        import uuid as _uuid
        from memory_system.semantic_store import Fact

        f = Fact(id=f"fact-{_uuid.uuid4().hex[:10]}", text=text.strip(),
                 entities=[e for e in (entities or []) if e], ts=time.time())
        with self._sleep_lock:
            kept = self.sleep._merge_into_existing([f])
            if not kept:
                return {"stored": False, "reason": "duplicate of existing fact"}
            deprecated = self.sleep._supersession_pass(kept)
            self.semantic.add_facts(kept)
            for fid in deprecated:
                self.semantic.deprecate(fid)
        return {"stored": True, "id": f.id, "superseded": deprecated}

    def build_messages(self, session_id: str, user_msg: str) -> tuple[list[dict], list[dict]]:
        """Assemble [system + short-term window + new message] and return the
        memories used (for transparency in the response)."""
        rec = self.recall(user_msg)
        used = rec["memories"]
        system = config.BASE_SYSTEM_PROMPT
        if rec["prompt_block"]:
            system += "\n\n" + rec["prompt_block"]

        messages: list[dict] = [{"role": "system", "content": system}]
        for t in self.episodic.recent_turns(session_id, config.SHORT_TERM_TURNS):
            messages.append({"role": t.role, "content": t.content})
        messages.append({"role": "user", "content": user_msg})
        return messages, used

    def log_turn(self, session_id: str, role: str, content: str) -> None:
        self.episodic.add_turn(session_id, role, content)

    # ----------------------------------------------------------------- sleep
    def consolidate(self, force: bool = False) -> dict:
        """Run a sleep cycle over closed sessions. force=True treats the
        active session as closed too (manual trigger)."""
        if not self._sleep_lock.acquire(blocking=False):
            return {"skipped": "consolidation already running"}
        try:
            idle = 0 if force else config.IDLE_CONSOLIDATE_S
            sessions = self.episodic.closed_sessions(idle_s=idle)
            stats = self.sleep.run(sessions)
            stats["at"] = time.time()
            self.last_sleep = stats
            log.info("sleep: %s", stats)
            return stats
        finally:
            self._sleep_lock.release()

    # ------------------------------------------------------------ inspection
    def list_memories(self, query: str | None = None, limit: int = 50) -> list[dict]:
        if query:
            res = self.retriever.retrieve(query, top_k=min(limit, 20))
            facts = [self.semantic.facts[f] for f in res.fact_ids if f in self.semantic.facts]
        else:
            facts = sorted(self.semantic.active_facts(), key=lambda f: -f.ts)[:limit]
        return [
            {"id": f.id, "text": f.text, "entities": f.entities, "ts": f.ts,
             "status": f.status}
            for f in facts
        ]

    def forget(self, fact_id: str) -> bool:
        if fact_id in self.semantic.facts:
            self.semantic.deprecate(fact_id)
            return True
        return False

    def stats(self) -> dict:
        return {
            "active_facts": len(self.semantic.active_facts()),
            "total_facts": len(self.semantic.facts),
            "turns_logged": len(self.episodic.turns),
            "raw_bytes": self.episodic.raw_bytes(),
            "active_bytes": self.semantic.active_bytes(),
            "last_sleep": self.last_sleep,
            "knobs": self.knobs.__dict__,
        }
