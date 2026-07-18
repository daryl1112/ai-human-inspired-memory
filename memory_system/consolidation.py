"""Sleep phase: engram maturation.

Pipeline per sleep cycle:
  1. Pull unconsolidated episodes from the episodic store.
  2. LLM-extract atomic facts (with entities) from each episode batch.
  3. Greedy embedding clustering -> merge near-duplicate facts.
  4. Supersession pass: new fact vs similar existing facts -> LLM decides
     'supersedes / distinct'; superseded facts are deprecated (forgotten
     from the active index, archived for audit).
  5. Write survivors to the semantic store; mark episodes consolidated.
"""
from __future__ import annotations

import datetime
import logging
import uuid

import numpy as np

import config
from config import Knobs
from memory_system.episodic_store import EpisodicStore
from memory_system.ollama_client import OllamaClient
from memory_system.semantic_store import Fact, SemanticStore

log = logging.getLogger("sleep")


def _normalize_extraction(out) -> list[dict]:
    """The model may return {'facts': [...]}, a bare list, or lists containing
    strings / partial dicts. Coerce everything into well-formed items and
    silently drop garbage — a malformed episode must never kill a run."""
    if isinstance(out, dict):
        out = out.get("facts", out.get("items", []))
    if not isinstance(out, list):
        return []
    items: list[dict] = []
    for it in out:
        if isinstance(it, str) and it.strip():
            items.append({"fact": it.strip(), "entities": [], "kind": "decision"})
        elif isinstance(it, dict) and isinstance(it.get("fact"), str) and it["fact"].strip():
            ents = it.get("entities", [])
            items.append({
                "fact": it["fact"].strip(),
                "entities": [str(e) for e in ents if e] if isinstance(ents, list) else [],
                "kind": it.get("kind", "decision"),
            })
    return items

_EXTRACT_SYS = (
    "You extract durable knowledge from a chat session between a user and "
    "their AI assistant: the user facts, decisions, preferences, projects, "
    "and corrections of earlier statements. "
    'Return strict JSON: {"facts": [{"fact": str, "entities": [str], '
    '"kind": "rule|decision|fix|noise"}]}. '
    "A fact is one atomic, self-contained sentence. Preserve any ticket codes, "
    "rule IDs, version numbers, or named systems verbatim inside the fact. "
    "Mark transient chatter, failed attempts later abandoned, and routine "
    "debugging churn as kind=noise. Corrections/replacements of earlier "
    "decisions are kind=decision."
)

_SUPERSEDE_SYS = (
    "You maintain a knowledge base. Given a NEW fact and an OLD fact, decide "
    "if the NEW one supersedes/replaces/corrects the OLD one (same subject, "
    "updated truth), or if they are distinct facts that can coexist. "
    'Return strict JSON: {"verdict": "supersedes" | "distinct"}.'
)


class SleepPhase:
    def __init__(self, ollama: OllamaClient, episodic: EpisodicStore,
                 semantic: SemanticStore, knobs: Knobs):
        self.ollama = ollama
        self.episodic = episodic
        self.semantic = semantic
        self.knobs = knobs

    # ------------------------------------------------------------------ main
    def run(self, sessions: list[tuple[str, str, float]]) -> dict:
        """sessions: (session_id, transcript_text, last_ts) from
        EpisodicStore.closed_sessions()."""
        if not sessions:
            return {"sessions": 0, "facts_added": 0, "facts_deprecated": 0}

        candidates: list[Fact] = []
        for sid, text, ts in sessions:
            for item in self._extract(sid, text, ts):
                if item.get("kind") == "noise":
                    continue  # forgetting starts at encoding
                candidates.append(
                    Fact(
                        id=f"fact-{uuid.uuid4().hex[:10]}",
                        text=item["fact"].strip(),
                        entities=[e for e in item.get("entities", []) if e],
                        ts=ts,
                        source_episodes=[sid],
                    )
                )

        merged = self._dedupe(candidates)
        merged = self._merge_into_existing(merged)  # cross-cycle dedup
        deprecated = self._supersession_pass(merged)
        self.semantic.add_facts(merged)
        # deprecate AFTER adding so new facts that supersede other new facts work
        for fid in deprecated:
            self.semantic.deprecate(fid)
        self.episodic.mark_consolidated({sid for sid, _, _ in sessions})
        stats = {
            "sessions": len(sessions),
            "facts_added": len(merged),
            "facts_deprecated": len(deprecated),
        }
        log.info("sleep cycle: %s", stats)
        return stats

    # ----------------------------------------------------------------- steps
    def _extract(self, sid: str, text: str, ts: float) -> list[dict]:
        when = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        try:
            out = self.ollama.chat_json(
                f"CHAT SESSION ({when}):\n{text}",
                system=_EXTRACT_SYS,
                model=config.EXTRACT_MODEL,
                num_ctx=8192,
                temperature=0.0,
                num_predict=3000,
            )
        except ValueError:
            log.warning("extraction JSON failure on session %s; skipping", sid)
            return []
        return _normalize_extraction(out)

    def _dedupe(self, facts: list[Fact]) -> list[Fact]:
        """Greedy cosine clustering; keep the NEWEST fact per cluster
        (tie-break: longest) and union entities/sources. Recency-weighted
        reconsolidation: if a correction clusters with the stale fact it
        replaces, the newer truth must be the survivor."""
        if len(facts) < 2:
            return facts
        vecs = np.array(self.ollama.embed([f.text for f in facts]))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        kept: list[int] = []
        assign: dict[int, int] = {}
        for i in range(len(facts)):
            placed = False
            for k in kept:
                if float(vecs[i] @ vecs[k]) >= self.knobs.cluster_sim_threshold:
                    assign[i] = k
                    placed = True
                    break
            if not placed:
                kept.append(i)
                assign[i] = i
        out: dict[int, Fact] = {}
        for i, k in assign.items():
            cand = facts[i]
            if k not in out:
                out[k] = cand
                continue
            rep = out[k]
            newer, older = (
                (cand, rep)
                if (cand.ts, len(cand.text)) > (rep.ts, len(rep.text))
                else (rep, cand)
            )
            newer.entities = sorted(set(newer.entities + older.entities))
            newer.source_episodes = sorted(
                set(newer.source_episodes + older.source_episodes)
            )
            out[k] = newer
        return list(out.values())

    _NEAR_DUP_SIM = 0.92
    _MAX_SUPERSEDE_CHECKS = 4  # LLM-call cap per new fact

    def _merge_into_existing(self, new_facts: list[Fact]) -> list[Fact]:
        """Cross-cycle dedup: a new fact near-identical to an existing active
        fact is absorbed into it (sources unioned) instead of re-added —
        repetition across days must compress, not accumulate."""
        existing = self.semantic.active_facts()
        if not existing or not new_facts:
            return new_facts
        ev = np.array(self.ollama.embed([f.text for f in existing]))
        ev /= np.linalg.norm(ev, axis=1, keepdims=True) + 1e-9
        nv = np.array(self.ollama.embed([f.text for f in new_facts]))
        nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9
        keep: list[Fact] = []
        for i, nf in enumerate(new_facts):
            sims = ev @ nv[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= self._NEAR_DUP_SIM:
                ex = existing[j]
                ex.source_episodes = sorted(set(ex.source_episodes + nf.source_episodes))
                ex.entities = sorted(set(ex.entities + nf.entities))
            else:
                keep.append(nf)
        return keep

    def _supersession_pass(self, new_facts: list[Fact]) -> list[str]:
        """Compare each new fact against candidate existing/earlier facts.
        Candidate gate: cosine >= threshold OR shared entity (corrections are
        often phrased very differently from what they replace). Newer wins
        on a 'supersedes' verdict; LLM calls capped per new fact."""
        deprecated: list[str] = []
        existing = self.semantic.active_facts()
        pool = existing + new_facts
        if len(pool) < 2:
            return deprecated
        vecs = np.array(self.ollama.embed([f.text for f in pool]))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        n_exist = len(existing)
        for i, nf in enumerate(new_facts):
            vi = vecs[n_exist + i]
            nf_ents = {e.lower() for e in nf.entities}
            candidates: list[tuple[float, Fact]] = []
            for j, old in enumerate(pool[: n_exist + i]):
                if old.ts > nf.ts:
                    continue
                sim = float(vi @ vecs[j])
                ent_hit = bool(nf_ents & {e.lower() for e in old.entities})
                if sim >= self.knobs.contradiction_sim_threshold or ent_hit:
                    candidates.append((sim, old))
            candidates.sort(key=lambda x: -x[0])
            for sim, old in candidates[: self._MAX_SUPERSEDE_CHECKS]:
                if old.id in deprecated:
                    continue
                if self._ask_supersedes(nf.text, old.text) == "supersedes":
                    deprecated.append(old.id)
        return deprecated

    def _ask_supersedes(self, new: str, old: str) -> str:
        try:
            out = self.ollama.chat_json(
                f"NEW fact: {new}\nOLD fact: {old}",
                system=_SUPERSEDE_SYS,
                model=config.EXTRACT_MODEL,
                num_ctx=2048,
                temperature=0.0,
                num_predict=256,
            )
            return out.get("verdict", "distinct") if isinstance(out, dict) else "distinct"
        except ValueError:
            return "distinct"
