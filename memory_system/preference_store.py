"""Procedural/preference memory ('Developer Journal').

Chosen/rejected pairs embedded by their situation description; at inference
time the top-N most relevant pairs are injected as few-shot exemplars —
pure in-context preference alignment, no fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Knobs
from memory_system.ollama_client import OllamaClient


@dataclass
class PreferencePair:
    id: str
    situation: str
    chosen: str
    rejected: str
    principle: str


class PreferenceStore:
    def __init__(self, ollama: OllamaClient, knobs: Knobs):
        self.ollama = ollama
        self.knobs = knobs
        self.pairs: list[PreferencePair] = []
        self._vecs: np.ndarray | None = None

    def load(self, pairs: list[PreferencePair]) -> None:
        self.pairs = pairs
        vecs = np.array(self.ollama.embed([p.situation for p in pairs]))
        self._vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def relevant(self, situation: str) -> list[PreferencePair]:
        if not self.pairs:
            return []
        q = np.array(self.ollama.embed([situation])[0])
        q /= np.linalg.norm(q) + 1e-9
        sims = self._vecs @ q
        order = np.argsort(-sims)[: self.knobs.n_preference_exemplars]
        return [self.pairs[i] for i in order]

    @staticmethod
    def as_prompt(pairs: list[PreferencePair]) -> str:
        blocks = []
        for p in pairs:
            blocks.append(
                f"### Situation\n{p.situation}\n"
                f"### My preferred approach (FOLLOW THIS STYLE)\n{p.chosen}\n"
                f"### Approach I rejected (NEVER do this)\n{p.rejected}\n"
                f"### Principle\n{p.principle}"
            )
        return "\n\n".join(blocks)
