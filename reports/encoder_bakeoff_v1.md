# Encoder bake-off v1

Metric: ranking accuracy (related > unrelated), n=25 reviewed triples (hard_within_trial=14, easy_cross_trial=11).
Tie-breaker between equal overall scores: hard-negative ranking accuracy.

| id | model | overall | hard | easy | dim |
|---|---|---|---|---|---|
| mpnet | `sentence-transformers/all-mpnet-base-v2` | 0.8400 | 0.7857 | 0.9091 | 768 |
| bge_large | `BAAI/bge-large-en-v1.5` | 0.8400 | 0.7857 | 0.9091 | 1024 |
| e5_large | `intfloat/e5-large-v2` | 0.8000 | 0.7143 | 0.9091 | 1024 |
| mini_baseline | `sentence-transformers/all-MiniLM-L6-v2` | 0.7600 | 0.7143 | 0.8182 | 384 |

**Winner (bake-off sort):** `mpnet` — tied with `bge_large` on overall+hard.

**Frozen TextEncoderV1 (owner override M2-A3):** `bge_large` —
`BAAI/bge-large-en-v1.5` (overall=0.8400, hard=0.7857, response_mark_scheme=0.8182).
Tie broken on highest-stakes sub-score, not a decisive bake-off win.
See `reports/DECISIONS.md` M2-A3.

