# Phase 1 milestone reports

One short entry per completed milestone: what was built, acceptance evidence, QC numbers of note.

---

## M0 — Skeleton, schemas, fixtures (2026-07-17)

**Built**
- Repo layout for Stage 1 (`configs/`, `schemas/`, `src/{data,utils}/`, `scripts/`, `tests/`, `fixtures/`, `legacy/`, `reports/`, `artifacts/`).
- JSON Schemas: `segment.json`, `fixation.json`, `star_conditions.json`.
- UTF-8 I/O + schema validators in `src/utils/io.py`.
- Config placeholders: `configs/data.yaml`, `configs/preprocessing.yaml`.
- Two synthetic fixture trials (`fx01_T99`, `fx02_T98_star_on`): 10 segments + 40 fixations each; empty-space fixations; expected edges covering NEXT/PREV/BELONGS_TO/SPATIAL/SEMANTIC; multi-relation pair `seg_r2`–`seg_ms1`.
- Preconditions: git init; `legacy/gaze-feature-engineering.py` copied; conda env `gnn-gaze` (Python 3.11); CUDA torch `2.6.0+cu124` (RTX 3080 Ti); `torch-geometric 2.8.0`; `requirements-lock.txt` frozen.
- PRE-1 confirmed: `T10-complete.json` present (typo file gone).

**Accept**
- `pytest`: 12 passed (schema validators + both fixtures load through every schema).
- `torch.cuda.is_available()` → `True`.
