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

---

## P2 — Metadata compilation (2026-07-17)

**Built**
- `src/data/segments.py`: box-union geometry; canonical panel map (UI → schema `ui`); panel-region table; P2.7 fallbacks (`segment_role`, spatial `aoi_id`, `segment_order` tie-break); empty strings → null; `star_chart_annotations` ignored.
- Schema `segment.json` extended for geometry AABB, QC fields, retained `aoi_type`/`box_ids`/`fallbacks_applied`.
- `scripts/run_p2_metadata.py` (runs P2.6 audit gate then compile); `tests/test_segments.py`.
- Outputs: `data_processed/v0_p0/metadata/` (`*__segments.json` + parquet companion, `*__panels.*`, `p2_summary.json`).

**Accept**
- P2.6 audit: **36/36 PASS** (exit 0).
- Compile: **36/36** variants; schema validation clean; `n_unclaimed_boxes_total=0`.
- Fallback totals: `segment_role_derived=1206`, `segment_order_tiebreak=168` (no spatial `aoi_id` needed).
- `pytest`: 35 passed (includes geometry/panel/fallback unit tests + updated fixtures).

---

## P3 — AOI hit injection (2026-07-17)

**Built**
- `src/data/aoi_injection.py`: star-chart override (star_on only) + additive UI one-hots/labels; smaller-region label priority; columns on all episodes.
- `scripts/run_p3_aoi_injection.py`; `tests/test_aoi_injection.py`.
- Canonical outputs: `data_processed/v0_p0/gaze_canonical/pXX.parquet` **and** `pXX.tsv` (UTF-8 companions; parquet remains pipeline-canonical) (+ `injection_qc.*`, `p3_summary.json`). Downstream P4–P7 read parquet, not P1 pruned.

**Accept**
- Unit tests: inside/outside/boundary; UI never overrides content labels; star overrides commentary; star_off untouched — green.
- Real data: 25 participants, **750** episodes, **75** star_on; star hits/relabels **408526**; UI hits answer_scroll_bar **59470**, commentary_scroll_bar **21217**, general_ui **115006** (scrollbar rates indicative).
- `p3_summary.json` `ok: true`.

---

## P4 — Visual Gate 1 (2026-07-17) — STOP for owner sign-off

**Built**
- `src/viz/overlay_check.py` + `scripts/gaze_overlay_check.py`: Plotly self-contained HTML overlays (document image, segment/panel/UI/star outlines, sample↔fixation toggle, time play/slider, AOI_label colours incl. P3 UI/star, injection QC sidebar, alignment %).
- `tests/test_overlay_smoke.py`; config `gate1` in `configs/preprocessing.yaml`.

**Accept (tooling)**
- Smoke HTML from fixtures: all required panels present (`pytest` + `--smoke` green).
- Stratified batch: **75** reports, **25** participants × ≥3 trials, star_on covers T11/T12/T13/T21/T27/T30; P2 audit had 0 ERROR flags to add.
- Index: `reports/gaze_checks/gate1/index.html` (generated HTMLs gitignored — regenerate with `python scripts/gaze_overlay_check.py`).

**STOP:** Owner reviews the stratified sample and records sign-off in `reports/DECISIONS.md`. **P5 must not start until that entry exists.** Do not self-certify.

---

## P5 — Coordinate finalisation (2026-07-17)

**Built**
- `src/data/coords.py`: DOCnorm (`x_docnorm`, `y_docnorm`) from P0 `W_doc`/`H_doc`; isotropic switch; viewport features (`y_screen`, `viewport_doc_position`, `gaze_viewport_y`); `H_screen_px=1080` in config.
- `scripts/run_p5_coords.py`; `tests/test_coords.py` (hand-computed + real episode).
- Outputs: `data_processed/v0_p0/gaze_coords/` (parquet + TSV; raw doc coords preserved).

**Accept**
- Unit tests green; 25/25 participants enriched (**16 179 675** rows); required columns present; `p5_summary.json` `ok: true`.
- P4 sign-off recorded in `DECISIONS.md` (Peter Andrews, 2026-07-17).
- Note: assignment still uses raw `gaze_point_*_doc`; DOCnorm/viewport feed P6 only.
