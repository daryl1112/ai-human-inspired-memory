"""Episodic store: append-only conversation log (the 'hippocampus').

Stores individual chat turns with real timestamps, grouped by session.
Sleep consolidation consumes whole *closed* sessions (idle long enough) and
marks them consolidated; raw turns stay on disk as the audit trail.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Turn:
    session_id: str
    role: str            # user | assistant
    content: str
    ts: float
    consolidated: bool = False


class EpisodicStore:
    def __init__(self, path: Path):
        self.path = path
        self.turns: list[Turn] = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    self.turns.append(Turn(**json.loads(line)))

    # ----------------------------------------------------------------- write
    def add_turn(self, session_id: str, role: str, content: str,
                 ts: float | None = None) -> Turn:
        t = Turn(session_id=session_id, role=role, content=content,
                 ts=ts if ts is not None else time.time())
        self.turns.append(t)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(t)) + "\n")
        return t

    def mark_consolidated(self, session_ids: set[str]) -> None:
        for t in self.turns:
            if t.session_id in session_ids:
                t.consolidated = True
        with self.path.open("w") as f:
            for t in self.turns:
                f.write(json.dumps(asdict(t)) + "\n")

    # ------------------------------------------------------------------ read
    def recent_turns(self, session_id: str, n: int) -> list[Turn]:
        return [t for t in self.turns if t.session_id == session_id][-n:]

    def last_activity(self) -> float:
        return max((t.ts for t in self.turns), default=0.0)

    def closed_sessions(self, idle_s: float, now: float | None = None
                        ) -> list[tuple[str, str, float]]:
        """Sessions with unconsolidated turns whose last turn is at least
        idle_s old. Returns (session_id, transcript_text, last_ts) oldest
        first - the unit of consolidation."""
        now = now if now is not None else time.time()
        by_sid: dict[str, list[Turn]] = {}
        for t in self.turns:
            if not t.consolidated:
                by_sid.setdefault(t.session_id, []).append(t)
        out = []
        for sid, turns in by_sid.items():
            last = max(t.ts for t in turns)
            if now - last >= idle_s:
                text = "\n".join(f"{t.role}: {t.content}" for t in turns)
                out.append((sid, text, last))
        return sorted(out, key=lambda x: x[2])

    def raw_bytes(self) -> int:
        return sum(len(t.content.encode()) for t in self.turns)
