"""Working memory (Baddeley–Hitch): a central executive that scans the
haystack chunk-by-chunk, admitting only salient rules/facts into a bounded
scratchpad (phonological loop), then reasons over the scratchpad alone.

The haystack never enters the answering context — that is the point.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Knobs
from memory_system.ollama_client import OllamaClient

_SCAN_SYS = (
    "You are a precision extraction module. From the log chunk provided, "
    "extract ONLY unusual, specific, idiosyncratic rules, constraints, or "
    "policies (things a team explicitly decided, named rules, odd exceptions). "
    "Ignore routine log noise, timestamps, generic chatter, and boilerplate. "
    "Output one rule per line, verbatim where possible. If none, output NONE."
)

_ANSWER_SYS = (
    "You answer strictly and only from the provided scratchpad notes. "
    "If a note contains a rule that applies, follow it exactly. "
    "Answer concisely."
)


@dataclass
class WorkingMemoryResult:
    answer: str
    scratchpad: str
    scratchpad_tokens: int
    haystack_tokens: int
    chunks_scanned: int


def _approx_tokens(text: str) -> int:
    # Log lines (timestamps, IDs, numbers) tokenize denser than prose:
    # ~3 chars/token, not 4. Underestimating here caused silent num_ctx
    # truncation ("truncated = 1") and runaway generations.
    return max(1, len(text) // 3)


class WorkingMemory:
    def __init__(self, ollama: OllamaClient, knobs: Knobs):
        self.ollama = ollama
        self.knobs = knobs

    def solve(self, haystack: str, question: str) -> WorkingMemoryResult:
        chunks = self._chunk(haystack, self.knobs.chunk_chars)
        notes: list[str] = []
        for chunk in chunks:
            hint = (
                f"A question will later be asked on this topic: \"{question}\"\n"
                "Prioritize rules relevant to it, but extract any idiosyncratic "
                "rule you find.\n\nLOG CHUNK:\n"
            )
            out = self.ollama.chat(
                hint + chunk,
                system=_SCAN_SYS,
                num_ctx=min(16384, max(8192, _approx_tokens(chunk) + 2048)),
                temperature=0.0,
                num_predict=800,
            )
            for line in out.splitlines():
                line = line.strip(" -*\t")
                if line and line.upper() != "NONE":
                    notes.append(line)
            notes = self._compress_if_needed(notes, question)

        scratchpad = "\n".join(f"- {n}" for n in notes) or "- (no rules found)"
        answer = self.ollama.chat(
            f"SCRATCHPAD NOTES:\n{scratchpad}\n\nQUESTION: {question}",
            system=_ANSWER_SYS,
            num_ctx=4096,
            temperature=0.0,
            num_predict=400,
        )
        return WorkingMemoryResult(
            answer=answer,
            scratchpad=scratchpad,
            scratchpad_tokens=_approx_tokens(scratchpad),
            haystack_tokens=_approx_tokens(haystack),
            chunks_scanned=len(chunks),
        )

    # ------------------------------------------------------------------ util
    @staticmethod
    def _chunk(text: str, size: int) -> list[str]:
        out, i = [], 0
        while i < len(text):
            j = min(len(text), i + size)
            if j < len(text):  # break on a line boundary when possible
                nl = text.rfind("\n", i, j)
                if nl > i + size // 2:
                    j = nl
            out.append(text[i:j])
            i = j
        return out

    def _compress_if_needed(self, notes: list[str], question: str) -> list[str]:
        budget = self.knobs.scratchpad_budget_tokens
        joined = "\n".join(notes)
        if _approx_tokens(joined) <= budget:
            return notes
        # Central-executive gating: keep only what plausibly matters.
        out = self.ollama.chat(
            "Prune this note list to fit a tight budget. Keep every note that "
            f"could bear on the question \"{question}\" and any named/numbered "
            "rules; drop duplicates and generic noise. Output the surviving "
            f"notes only, one per line, max ~{budget} tokens total.\n\n" + joined,
            num_ctx=8192,
            temperature=0.0,
            num_predict=budget + 200,
        )
        pruned = [l.strip(" -*\t") for l in out.splitlines() if l.strip()]
        return pruned or notes[: max(3, len(notes) // 2)]
