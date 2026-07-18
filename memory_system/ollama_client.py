"""Thin Ollama HTTP client with retries, timing, and model management."""
from __future__ import annotations

import json
import logging
import time

import requests

import config

log = logging.getLogger("ollama")

_TIMEOUT = 900  # iGPU prompt-processing on big chunks can be slow


class OllamaClient:
    def __init__(self, base_url: str = config.OLLAMA_URL):
        self.base = base_url.rstrip("/")
        self.call_count = 0
        self.total_latency = 0.0

    # ---------------------------------------------------------------- health
    def wait_ready(self, timeout_s: int = 300) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base}/api/tags", timeout=5)
                if r.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise RuntimeError(f"Ollama not reachable at {self.base}")

    def ensure_model(self, name: str) -> None:
        tags = requests.get(f"{self.base}/api/tags", timeout=30).json()
        present = {m["name"] for m in tags.get("models", [])}
        if any(p == name or p.split(":")[0] == name for p in present):
            return
        log.info("Pulling model %s (this can take a while)...", name)
        with requests.post(
            f"{self.base}/api/pull", json={"name": name}, stream=True, timeout=None
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    status = json.loads(line).get("status", "")
                    if "success" in status:
                        log.info("Model %s ready.", name)

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        model: str = config.WORKER_MODEL,
        num_ctx: int = 8192,
        temperature: float = 0.1,
        max_retries: int = 3,
        num_predict: int = 1024,
        fmt: str | None = None,
    ) -> str:
        messages = []
        # '/no_think' covers Ollama versions that ignore the 'think' flag —
        # without it Qwen3 can emit unbounded <think> blocks and time out.
        sys_prompt = ((system + "\n") if system else "") + "/no_think"
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,  # disable qwen3 thinking blocks for speed
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "num_predict": num_predict,  # hard cap: generation can't run away
            },
        }
        if fmt:
            body["format"] = fmt  # Ollama structured output: prose is impossible
        last_err: Exception | None = None
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                r = requests.post(f"{self.base}/api/chat", json=body, timeout=_TIMEOUT)
                r.raise_for_status()
                dt = time.time() - t0
                self.call_count += 1
                self.total_latency += dt
                content = r.json()["message"]["content"]
                return _strip_think(content)
            except requests.RequestException as e:  # transient GPU hiccups
                last_err = e
                log.warning("chat attempt %d failed: %s", attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Ollama chat failed after retries: {last_err}")

    def chat_json(self, prompt: str, system: str | None = None, **kw) -> dict | list:
        """Chat with format=json constrained decoding, one repair retry."""
        kw.setdefault("fmt", "json")
        raw = self.chat(prompt, system=system, **kw)
        for attempt in range(2):
            try:
                return _extract_json(raw)
            except ValueError:
                if attempt == 0:
                    raw = self.chat(
                        "Reformat the following as strict JSON only, no prose, "
                        "no markdown fences:\n\n" + raw,
                        num_ctx=8192,
                        fmt="json",
                    )
        raise ValueError(f"Could not parse JSON from model output: {raw[:400]}")


    # ---------------------------------------------------------- streaming chat
    async def chat_stream(self, messages: list[dict],
                          model: str = config.WORKER_MODEL,
                          num_ctx: int = 8192,
                          temperature: float = 0.4,
                          num_predict: int = 1500):
        """Async generator yielding content deltas from /api/chat."""
        import httpx

        sys_seen = any(m["role"] == "system" for m in messages)
        if sys_seen:
            messages = [
                {**m, "content": m["content"] + "\n/no_think"}
                if m["role"] == "system" else m
                for m in messages
            ]
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature,
                        "num_predict": num_predict},
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
            async with client.stream("POST", f"{self.base}/api/chat", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        self.call_count += 1

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], model: str = config.EMBED_MODEL) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):  # batch to keep requests small
            batch = texts[i : i + 32]
            r = requests.post(
                f"{self.base}/api/embed",
                json={"model": model, "input": batch},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks if the model emits them anyway."""
    while "<think>" in text and "</think>" in text:
        a = text.index("<think>")
        b = text.index("</think>") + len("</think>")
        text = text[:a] + text[b:]
    return text.strip()


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not start_candidates:
        raise ValueError("no JSON found")
    start = min(start_candidates)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj
