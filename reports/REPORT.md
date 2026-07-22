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

---

## P6 — Fixation construction (2026-07-17)

**Built**
- `src/data/fixations.py`: legacy-compatible event aggregation; doc-space fixation medians + DOCnorm; saccade→`prev_saccade`; scroll features; orchestrates assignment + loops.
- `src/data/gaze_assignment.py`: ε-dilated assignment policy + empty-space (smaller region wins).
- `src/data/loops.py`: visit/return + A→B→A template detector.
- `src/data/epsilon.py`: corpus ε from P1 `epsilon_inputs` → **27.093 px** (no fallback).
- Tests: `test_fixations`, `test_gaze_assignment`, `test_loops`; `scripts/run_p6_fixations.py`.

**Accept**
- 750/750 episodes; 0 errors; schema sample validation green.
- ε sensitivity mean % changed: ×0.5 → **5.84%**, ×1.5 → **3.85%**.
- Empty-space ~**14.0%**; ambiguous ~**13.7%**; mean confidence ~**0.45**.
- All 6 loop templates ≥50 corpus count (none dropped for D2).
- Outputs: `data_processed/v0_p0/fixations/{Pxx}/{trial}__{star}.parquet` **and** `.tsv` (parquet pipeline-canonical) + QC/sensitivity tables.

---

## P7 — Visual Gate 2 (2026-07-17) — HARD STOP for owner sign-off

**Built**
- Extended `scripts/gaze_overlay_check.py --gate 2` via `src/viz/gate2_overlay.py`: assigned/empty/ambiguous layers, current-fixation highlight + ε rings + alt boxes, assignment QC sidebar, distance-to-edge histogram, panel vs export-AOI counts.
- Sample = Gate 1 stratified set **plus** P6 QC flags (empty>40% / ambig>40% / conf<0.2).
- Tests: `tests/test_gate2_overlay.py`.

**Accept (tooling)**
- Smoke HTML green; batch **117** reports (`n_flagged_qc_added=46`); star_on covers all 6 eligible trials; `reports/gaze_checks/gate2/index.html`.

**STOP:** Owner reviews Gate 2 and records sign-off in `reports/DECISIONS.md`. **No Stage 2 / model code (M2+) until then.** Do not self-certify.

---

## Stage 1 consolidation (2026-07-20) — complete

P7 Gate 2 signed off (`reports/DECISIONS.md`). Stage 1 closed.

Pipeline: P0 registries → P1 prune → P2 metadata → P3 AOI → P4 Gate 1 → P5 coords → P6 fixations → P7 Gate 2.

**Unblocked:** Stage 2 / M2 (text encoder selection).

---

## M2 — Text encoder selection (2026-07-20) — complete

**Built**
- Pair curation (M2-A1/A2): 25 reviewed triples (14 hard / 11 easy).
- Bake-off via HF mean-pool backend; report `reports/encoder_bakeoff_v1.md`.
- **TextEncoderV1 (M2-A3 override):** `BAAI/bge-large-en-v1.5` (dim=**1024**,
  revision `d4aa6901…`, mean pool, L2). Tie on overall+hard broken on
  `response_mark_scheme` (not a decisive bake-off win) — see DECISIONS.md.
- `configs/graph.yaml`: `text_embedding_dim: 1024`, `graph_version: g1_bge1024`.

**Accept**
- Bake-off table present; TextEncoderV1 card + revision + sha256 pinned.
- Unit tests green; promote gate passed (n=25 ≥ 20).

---

## M3 — Automatic graph parser (2026-07-20) — complete

**Built**
- `configs/graph.yaml`: `g1_bge1024`, `text_embedding_dim=1024` (TextEncoderV1 = bge).
- Edge builders + features + `build_graph_dict`; embedding cache; `scripts/run_m3_graphs.py`.
- **36/36** graphs: `data_processed/graphs/g1_bge1024/{trial}__{star}.pt`.
- Embeddings: `data_processed/embeddings/g1_bge1024/`.
- Diagnostics: `reports/graph_diagnostics/g1_bge1024/REPORT.md`.

**Edge totals (directed):** NEXT/PREV 1060 each; BELONGS_TO 1206; SPATIAL 4200; SEMANTIC 5153.

**NS↔S (M3-C1):** all 6 eligible trials **PASS** (0 missing); star-conditional S-only excluded per allowlist.

**Accept**
- Per-edge unit tests green; dim check vs card green; 36 graphs on disk; correspondence ok.

---

## M4 — Compact GNN, standalone (2026-07-20)

**Built**
- `configs/gnn.yaml`; `src/models/gnn.py`: pure-torch GATv2-style `CompactGNN`
  (relation embeddings + numeric edge attrs in attention, residuals, edge dropout);
  returns `x_v` (projected) and `h_v` (contextualised); attention extractable.
- Throwaway panel probe (`scripts/run_m4_panel_probe.py`): panel features masked from `x`;
  classify panel from `h_v` — mean acc **1.0** across seeds {13,42,1337} on 6 graphs
  (sanity only; model discarded).

**Accept**
- `pytest tests/test_gnn.py` green (shapes, grads, attention, panel probe, 3-seed stability).
- Panel-probe dry-run written to `runs/m4_panel_probe/panel_probe_summary.json`.

**Note:** Implemented without importing `torch_geometric` (Windows import crash via
pyarrow/PyG LLM extras). Layer is GATv2-style message passing as specified in §7.

---

## M5 — Fixation tokens and episode dataset (2026-07-20)

**Built**
- `configs/dataset.yaml`: empty-space mode (`panel_specific`), ranking 8 easy + 4 hard,
  scroll dropout documented (applied in M6), panel class list.
- `src/models/tokens.py`: side features (timing, saccade, scroll, visit/return, loop_role,
  assignment confidence) + `EmptySpaceEmbedding` + `assemble_token` → concat[`x_v`,`h_v`,side].
- `src/data/targets.py`: multi-hot next-relation over graph relations +
  `NO_DIRECT_RELATION` / `EMPTY_SPACE_TRANSITION`; ranking candidate sampler.
- `src/data/episode_dataset.py`: `EpisodeDataset` + `collate_episodes`; optional CompactGNN
  for `x_v`/`h_v` (tests use feature-slice placeholder); fixture + real loaders.
- `scripts/run_m5_relation_freq.py`: full-corpus label frequency table for M6 BCE weights.

**Accept**
- Fixture multi-hot unit tests (hand-computed NEXT, multi-relation, NO_DIRECT, empty-space)
  + fixture integration (q1→q2 NEXT; empty-space label; lookup consistency).
- Relation frequencies on **750** episodes / **218292** transitions
  (`reports/relation_frequencies_m5.md`): dominant `NO_DIRECT_RELATION` (0.62),
  then SPATIAL (0.23), EMPTY_SPACE (0.10), NEXT (0.10). `BELONGS_TO` = 0 on consecutive
  fixations (panel abstract nodes are not gaze targets — expected).
- Throughput: **45.4** real episodes/s (freq script); fixture Dataset ≥20 ep/s sanity.
- `pytest` green (incl. fixture multi-relation doc updated to NEXT+SPATIAL within-panel).

---

## M6 — Causal loop-aware transformer + three losses (2026-07-20)

**Built**
- `configs/model_transformer.yaml` (d_model=192, 4 layers, 4 heads) + `configs/splits.yaml`
  (grouped 5-fold participants) + relation weights already locked in `configs/train.yaml`.
- `src/models/{biases,transformer,heads}.py`: causal transformer with temporal /
  graph-relation / loop-return attention biases; three heads (panel, relation, ranking).
- `src/train/{losses,loop,sampling}.py`: exactly three losses (CE / weighted BCE /
  ranking CE); AdamW + grad clip + early stop; scroll-feature dropout p=0.3;
  run-dir logging (`metrics.jsonl`, configs, git hash, checkpoint).
- `src/data/splits.py`; `src/utils/tracking.py` (none/mlflow/wandb; never aborts run).
- Minimal V1+V2 HTML: `src/eval/viz/training.py` → `runs/*/viz/report.html`.
- Scripts: `scripts/run_m6_dryrun.py` (fixtures); `scripts/run_m6_train.py` (owner full runs).

**Accept (agent)**
- Causal-mask leakage test green (future token perturbation → past `y_t` unchanged).
- Fixture overfit path: loss drops >50% and below 1.5 on 2 fixtures (tiny model, 80 steps).
- BELONGS_TO excluded from active BCE labels; clipped weights from M5 table.
- Grouped folds: no participant leakage across train/val.
- Dry-run: `runs/m6_dryrun/` (15 epochs on 2 fixtures) + `viz/report.html`.
- `pytest` full suite green.

**Owner next:** full grouped 5-fold × seeds {13,42,1337} via `scripts/run_m6_train.py`
(not run by the agent). Cross-machine fixture-overfit identity check before trusting
uni-server full runs. Pip lockfile already present (`requirements-lock.txt`).

---

## M6 — Diagnostic go/no-go + matrix launch (2026-07-22)

**Predictive eval** on `runs/m6/fold0_seed13/checkpoint_best.pt` (best val 2.4496 @ ep 64):
- SEMANTIC_CANDIDATE AP **0.165** vs base-rate **0.037** → **GO**
  (`reports/m6_fold0_seed13_predictive_metrics.md`).
- Ranking MRR 0.789 (freq baseline 0.457; feature-cosine 0.789); panel acc 0.827 / macro-F1 0.778.
- Known: final train/val gap ~0.62 — no extra regularisation before matrix.

**Config lock:** `clip_max: 12`, `max_epochs: 100`, ES patience 10, `tracking.backend: mlflow`,
`ablation_id: baseline`. DECISIONS M6-W1 amendment + M6-W5.

**Matrix:** `scripts/run_m6_matrix.py` (5×3); V5 `src/eval/viz/compare_runs.py`.

---

## M7 — Diagnostic gate on fold0 seed13 best (2026-07-22)

**Built:** `src/eval/loop_diagnostics.py`, `scripts/run_m7_diagnostics.py`; return/loop
aux heads + losses for gate-fail retrain.

**Gate result (FAIL overall):** D1 margin 0.021 < 0.05; D2/D3 pass. Logged in
`reports/m7_fold0_seed13_diagnostics.md` + DECISIONS M7-G1. Remedy scoped to
**return_aux only** (loop_aux stays off — D2 passed). D2 zero-counts were a
re-detect bug (M7-G1b): LD template 2310 corpus-wide with role refinement;
star template 1001 corpus-wide.

---

## M6-W6 — Full-sequence validity lift (2026-07-22)

**Truncation QC** (`reports/truncation_analysis_m6w6.md`): at cap 256, **40.2%** of
fixations discarded; 43.1% of episodes truncated. Star-eligible median length 526
(89% truncated). Star loops: 64% abs≥256 and **75%** in second half of episode.

**M6-W2 was CPU AV**, not 16 GB OOM. Mem profile on 3080 Ti: T=1536×B=8 peaks
**2.85 GB** (`reports/mem_profile_seqlen.json`). Cap raised to **1536**;
`return_aux` unset pending re-gate. Full-seq train running under
`runs/m6_fullseq/fold0_seed13` (tag `baseline_fullseq_1536`). Prior cap-256
results superseded once this run completes (DECISIONS M6-W6).

---

## M6-W7 — Graph-relation bias restored (2026-07-22)

**Pre-flight:** T=1536×B=8 with bias ON peaks **3.20 GB**; four long-T GPU
correctness checks **PASS** at T=768 and 1536. Config: bias on, aux off, tag
`baseline_fullseq_1536_graphbias`, run root `runs/m6_fullseq_graphbias/`.
Bias-off fullseq run kept as reference only. Truncation counts now logged in
every `train_summary.json` / `run_meta.json`.
