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

---

## P0 — Registries and identity (2026-07-17)

**Built**
- `src/data/registry.py`: filename→identity parser; trial registry; document dimension registry (PIL); star-condition table from gaze `Star Chart`; `question_type`/`question_id` from gaze `Question type` (constant per trial); S/NS variant consistency check.
- `scripts/run_p0_registry.py`; tests in `tests/test_registry.py`.
- Outputs under `data_processed/v0_p0/registry/` (json + parquet).

**Accept**
- Unit tests: parser (T / TS / TNS + rejection), strip/star rules, hard vs soft vs triage variant cases — all green (with M0: 21 passed).
- Real data: 36 variants, 36 images, 750 star assignments, 30 question types; star-on = 3/participant validated.
- Variant segment asymmetries **triaged** in `reports/DECISIONS.md` P0-V1 (M3 base definition still needs owner choice).

---

## P1 — Gaze prune/tidy (2026-07-17)

**Built**
- `src/data/gaze_load.py`: Sensor filter, keep/drop lists, snake_case rename map, episode QC, DACSmm→ε input extraction before drop.
- `scripts/run_p1_gaze_prune.py`; `tests/test_gaze_load.py`.
- Outputs: `data_processed/v0_p0/gaze_pruned/pXX.parquet` + `episode_qc.parquet` + `epsilon_inputs.parquet`.

**Accept**
- Unit tests: keep/drop/rename column-by-column, correction_false counts, Trial Raw disagreement — green.
- Real data: all 25 participant TSVs pruned; QC appended to `reports/data_qc.md`.

---

## P0 rebuild + P2.6 re-audit + M3-C1 + P3-E1 (2026-07-17)

**Done**
- Document-dimension registry rebuilt after NS image/metadata fix.
- P2.6 audit: **36/36 PASS** (T11NS fixed; metadata path flattened to `_data/annotations-audited/complete`).
- NS↔S correspondence: **all six eligible trials PASS** (T11 star-instruction fragments allowlisted).
- M3-C1: per-variant construction + correspondence (amends frozen #11).
- P3-E1: AOI hit injection generalisation logged; PLAN/schema/config updated (UI additive hits; star-chart unchanged; P6 empty-space split).

**Unblocked:** P2 segment compilation can proceed. `T10-completee.json` typo still present (harmless to identity parser).
