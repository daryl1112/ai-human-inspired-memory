# memory-service

Standalone long-term memory for a chatbot, built from the benchmarked
human-like memory system (`best_knobs.json` from the tuning run is the
production retrieval config — drop it into `./data/`).

## Run

```bash
docker compose up --build        # ollama + qdrant + memory-service :8100
```

Already running ollama/qdrant from the benchmark stack? Remove those two
services from this compose file and point `OLLAMA_URL` / `QDRANT_URL` at
the existing containers.

## API

| Endpoint | What it does |
|---|---|
| `POST /recall` `{query, top_k?}` | `{"prompt_block", "memories"}` — ready-to-inject memory text |
| `POST /remember` `{session_id, role, content}` | log one turn (role: user\|assistant only) |
| `POST /memories` `{text, entities?}` | explicit immediate save; deduped + supersedes older facts |
| `POST /chat` `{message, session_id?}` | optional built-in SSE chat (API-only callers ignore this) |
| `POST /consolidate` | run a sleep cycle now (forces active sessions closed) |
| `GET /memories?q=` | list/search active facts |
| `DELETE /memories/{id}` | forget a fact (deprecates, keeps audit copy) |
| `GET /preferences` | current style-preference pairs (edit `data/preferences.json`) |
| `GET /stats`, `GET /health` | counters, last sleep, knobs in effect |

Consolidation also runs automatically: after 30 min of idle
(`IDLE_CONSOLIDATE_S`) for closed sessions, and nightly at 03:00
(`NIGHTLY_HOUR`) including the active session.

## Wiring an agent loop (API-only)

Three call sites in your `chat_stream`, mirroring how `lessons` already work:

```python
MEM = "http://memory-service:8100"

def _mem(path, payload):           # fire-and-forget; memory never breaks chat
    try:
        return httpx.post(f"{MEM}{path}", json=payload, timeout=5).json()
    except Exception:
        return {}

# 1. recall before the loop, next to reflect.lessons_for(...)
memory = _mem("/recall", {"query": text}).get("prompt_block", "")
messages = build_messages(context, lessons, memory)   # append like lessons

# 2. remember the user turn, next to create_message(..., "user")
_mem("/remember", {"session_id": str(conversation.id), "role": "user", "content": text})

# 3. remember the final reply, inside the no-tool-calls completion branch
_mem("/remember", {"session_id": str(conversation.id), "role": "assistant",
                   "content": msg.get("content", "")})
```

Don't log tool-call JSON or tool outputs — the final assistant message is the
distilled version; raw tool noise fights the sleep phase's noise filter.

Optional agent tools for the registry: `search_memory(query)` -> `GET
/memories?q=`, and `remember_fact(text)` -> `POST /memories` for deliberate,
immediate saves that skip the nightly wait.

## ChatAdapter side (only if using the built-in /chat)

Each `data:` event is JSON. Append `delta` fields as they arrive; the final
event carries `done: true` plus `session_id` (persist it — e.g.
localStorage — and send it with every subsequent message so short-term
context and session grouping work) and `memories_used`, which you can render
as a "remembered from <date>" chip under the reply.

## How memory flows

1. Every turn (user + assistant) is appended to `data/episodic.jsonl`.
2. Sleep cycles extract durable facts from closed sessions, dedupe them,
   and let newer decisions supersede older ones (superseded facts leave
   the active index but remain in Qdrant for audit — watch it live at
   `localhost:6333/dashboard`, collection `chat_memory_facts`).
3. On each message, hybrid retrieval (vectors + BM25 + entity graph +
   temporal cues like "last week") picks the top facts, which are injected
   into the system prompt with their dates, alongside matching preferences.
4. The service survives restarts: facts are rebuilt from Qdrant, turns
   from the JSONL log.

## Verify offline

```bash
python test_service.py     # 13 checks, no GPU needed
```
