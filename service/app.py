"""memory-service HTTP API.

POST /chat            SSE stream (optional convenience; API-only callers
                      can ignore it): {"delta"} chunks then {"done", ...}
POST /remember        log a conversation turn {session_id, role, content}
POST /recall          {"query"} -> {"prompt_block", "memories"}
POST /memories        explicit immediate save {"text", "entities?"}
POST /consolidate     run a sleep cycle now (force closes active sessions)
GET  /memories        list active facts, optional ?q= search
DELETE /memories/{id} deprecate (forget) one fact
GET  /preferences     current preference pairs
GET  /stats, /health

Background: idle watcher (consolidate after IDLE_CONSOLIDATE_S of silence)
and nightly consolidation at NIGHTLY_HOUR local time.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from service.manager import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-8s %(levelname)-5s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("api")

app = FastAPI(title="memory-service", version="1.0")
mm = MemoryManager()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class RememberRequest(BaseModel):
    session_id: str
    role: str  # user | assistant
    content: str


class RecallRequest(BaseModel):
    query: str
    top_k: int | None = None


class FactRequest(BaseModel):
    text: str
    entities: list[str] | None = None


@app.on_event("startup")
async def _startup() -> None:
    await asyncio.to_thread(mm.start)
    asyncio.create_task(_background_sleep_loop())
    log.info("memory-service ready")


async def _background_sleep_loop() -> None:
    """Idle trigger: pending turns + silence >= IDLE_CONSOLIDATE_S.
    Nightly trigger: first check on/after NIGHTLY_HOUR each day (forces
    even the still-active session through consolidation)."""
    last_nightly_date: datetime.date | None = None
    while True:
        await asyncio.sleep(60)
        try:
            now = datetime.datetime.now()
            nightly_due = (
                now.hour >= config.NIGHTLY_HOUR and last_nightly_date != now.date()
            )
            if nightly_due:
                stats = await asyncio.to_thread(mm.consolidate, True)
                if "skipped" not in stats:
                    last_nightly_date = now.date()
                    log.info("nightly consolidation: %s", stats)
                continue
            last = mm.episodic.last_activity()
            if last and time.time() - last >= config.IDLE_CONSOLIDATE_S:
                if mm.episodic.closed_sessions(config.IDLE_CONSOLIDATE_S):
                    stats = await asyncio.to_thread(mm.consolidate, False)
                    if stats.get("sessions"):
                        log.info("idle consolidation: %s", stats)
        except Exception:
            log.exception("background sleep loop error")


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    session_id = req.session_id or f"s-{uuid.uuid4().hex[:8]}"
    messages, used = await asyncio.to_thread(mm.build_messages, session_id, req.message)
    mm.log_turn(session_id, "user", req.message)

    async def event_stream():
        reply_parts: list[str] = []
        try:
            async for delta in mm.ollama.chat_stream(
                messages,
                num_ctx=config.CHAT_NUM_CTX,
                num_predict=config.CHAT_NUM_PREDICT,
            ):
                reply_parts.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        except Exception as e:  # surface errors as an SSE event, not a hang
            log.exception("stream failure")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        reply = "".join(reply_parts)
        if reply:
            mm.log_turn(session_id, "assistant", reply)
        done = {"done": True, "session_id": session_id, "memories_used": used}
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/remember")
async def remember(req: RememberRequest) -> dict:
    if req.role not in {"user", "assistant"}:
        raise HTTPException(422, "role must be 'user' or 'assistant'")
    if not req.content.strip():
        raise HTTPException(422, "content is empty")
    t = await asyncio.to_thread(
        mm.episodic.add_turn, req.session_id, req.role, req.content
    )
    return {"logged": True, "session_id": t.session_id, "ts": t.ts}


@app.post("/recall")
async def recall(req: RecallRequest) -> dict:
    return await asyncio.to_thread(mm.recall, req.query, req.top_k)


@app.post("/memories")
async def add_memory(req: FactRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(422, "text is empty")
    return await asyncio.to_thread(mm.remember_fact, req.text, req.entities)


@app.post("/consolidate")
async def consolidate() -> dict:
    return await asyncio.to_thread(mm.consolidate, True)


@app.get("/memories")
async def memories(q: str | None = None, limit: int = 50) -> list[dict]:
    return await asyncio.to_thread(mm.list_memories, q, limit)


@app.delete("/memories/{fact_id}")
async def forget(fact_id: str) -> dict:
    if not await asyncio.to_thread(mm.forget, fact_id):
        raise HTTPException(404, f"fact {fact_id} not found")
    return {"forgotten": fact_id}


@app.get("/preferences")
async def preferences() -> list[dict]:
    return [p.__dict__ for p in mm.prefs.pairs]


@app.get("/stats")
async def stats() -> dict:
    return mm.stats()


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "active_facts": len(mm.semantic.active_facts())}
