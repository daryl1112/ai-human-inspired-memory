# Human-Like AI Memory — Benchmark Results

Run completed: 2026-07-18 00:14:02 UTC

Worker model: `qwen3:30b-a3b` · Embeddings: `nomic-embed-text` · Seed: 1151 · Iterations run: 12

## Evaluation Dashboard

| Dimension | Target | Achieved | Met |
|---|---|---|---|
| Working Memory — needle retrieval accuracy | ≥ 95% | 1.000 | ✅ |
| Working Memory — context efficiency (scratchpad/haystack) | informational | 0.005 | ✅ |
| Long-Term Storage — storage cost reduction | ≥ 50% | 0.908 | ✅ |
| Long-Term Storage — forgetting accuracy | ≥ 90% | 0.789 | ❌ |
| Contextual Retrieval — recall@5 | ≥ 90% | 0.900 | ✅ |
| Contextual Retrieval — MRR | ≥ 0.60 | 0.825 | ✅ |
| Contextual Retrieval — avg latency (s) | informational | 0.028 | ✅ |
| Habit Retention — alignment score | ≥ 0.70 | 0.750 | ✅ |
| Habit Retention — avg cyclomatic complexity delta | < 0 | -0.133 | ✅ |
| Habit Retention — self-correction rate | informational | 0.519 | ✅ |

Secondary (LLM-judge, not gated): retrieval relevance 4.6/10, style fidelity 5.0/10.

## Summary

Not all targets were met within the iteration budget; the table above shows best-achieved values honestly. The system consolidates 50 raw session transcripts into an active semantic layer (vectors + knowledge graph), forgets superseded decisions via a contradiction-detection sleep phase, answers needle questions through a bounded working-memory scratchpad, and mirrors logged developer preferences with a linter-driven self-correction loop.

## Best hyperparameters

```json
{
  "chunk_chars": 8000,
  "scratchpad_budget_tokens": 600,
  "cluster_sim_threshold": 0.72,
  "contradiction_sim_threshold": 0.8,
  "top_k_dense": 20,
  "top_k_bm25": 10,
  "graph_hops": 1,
  "rrf_k": 60,
  "rerank": false,
  "rerank_pool": 12,
  "n_preference_exemplars": 3
}
```

## Iteration history

| # | Change | Aggregate | Targets met |
|---|---|---|---|
| 0 | baseline | 0.769 | no |
| 1 | chunk_chars=8000 | 0.799 | no |
| 2 | chunk_chars=16000 | 0.743 | no |
| 3 | chunk_chars=24000 | 0.605 | no |
| 4 | scratchpad_budget_tokens=400 | 0.718 | no |
| 5 | scratchpad_budget_tokens=900 | 0.680 | no |
| 6 | top_k_dense=5 | 0.942 | no |
| 7 | top_k_dense=10 | 0.916 | no |
| 8 | top_k_dense=20 | 0.982 | no |
| 9 | top_k_bm25=5 | 0.773 | no |
| 10 | graph_hops=2 | 0.748 | no |
| 11 | rrf_k=20 | 0.942 | no |

Full per-iteration metrics: `history.json`. Final code snapshot: `final_code/`.
