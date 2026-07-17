# PLAN.md — Phase 1 Execution Map

**Status:** approved planning artefact. This file maps every preprocessing stage (P0–P7) and milestone (M0–M9) to concrete tasks, files, and acceptance criteria. It is the execution companion to the three binding specification documents:

1. [docs/updated_gnn_transformer_gnn_llm_development_plan_v4.md](docs/updated_gnn_transformer_gnn_llm_development_plan_v4.md) — the research plan (architecture, losses, ablations, validation). Scientific source of truth.
2. [docs/phase1_dev_plan_cursor.md](docs/phase1_dev_plan_cursor.md) — implementation milestones M0–M9 with acceptance criteria.
3. [docs/gaze_preprocessing_plan.md](docs/gaze_preprocessing_plan.md) — data pipeline stages P0–P7. **Supersedes and expands milestones M1–M1b** of the dev plan.

If this file and a doc ever disagree, the docs win; log the discrepancy in `reports/DECISIONS.md`.

**Staging:**

- **Stage 1** = M0 (repo skeleton) + P0–P7 (full preprocessing pipeline, both visual gates).
- **Stage 2** = M2–M9 (encoder selection, graph construction, model build, training preparation). Training itself is run by the project owner afterwards.

Work **one milestone/stage at a time**, with passing tests, before moving on. The two visual gates (P4 and P7) end with the **owner's manual sign-off** — build the tooling, generate the review sample, then **stop and wait**.

---

## 0. Preconditions (owner + agent, before M0)

| # | Task | Who |
|---|------|-----|
| PRE-1 | Rename `_data\annotations-audited\complete\T10-completee.json` → `T10-complete.json` (filename typo) | **Owner** (agent must never modify `_data\`) |
| PRE-2 | Create the conda environment (see §1 below) and confirm `python -c "import torch; print(torch.cuda.is_available())"` prints `True` | Agent |
| PRE-3 | `git init` at the workspace root; commit `.gitignore` (ignoring `_data/`, `data_processed/`, `runs/`, conda/cache artefacts) before any code | Agent |
| PRE-4 | Copy `D:\Projects\EyeGaze\hierarchical-hmm-eye-tracking-analysis\data-processing\gaze-feature-engineering.py` into `legacy/gaze-feature-engineering.py` (read-only porting reference for P6) | Agent |

## 1. Environment (resolved decision)

- **Conda** environment, name `gnn-gaze`, Python **3.11** (per dev plan §1; the machine's system Python is 3.12.7 — do not use it directly).
- PyTorch ≥ 2.2 with CUDA build (RTX 3080 16 GB, driver CUDA 12.8 → use the cu121/cu124 wheel channel that PyTorch supports for the installed version), PyTorch Geometric, `sentence-transformers`, `scikit-learn`, `pandas`, `pyarrow`, `omegaconf`, `pytest`, `matplotlib`, `plotly`, `mlflow`, `jsonschema`, `pillow`, `statsmodels`.
- Commit `environment.yml` **and a pip lockfile** (`pip freeze` or `pip-compile` output): the torch/PyG install line differs per CUDA setup, so conda YAML alone underdetermines the env. Every dependency added later goes into both in the same commit. The lockfile is what makes the M6 cross-machine reproducibility check (laptop vs uni server) meaningful.

## 2. Data locations (read-only; never modify, never commit)

| Input | Location | Notes |
|---|---|---|
| Trial metadata (audited) | `_data\annotations-audited\complete\` | 36 JSON files: 24 non-eligible trials (`T01`…`T29`), 6 star-eligible trials with S/NS variants (`T11`, `T12`, `T13`, `T21`, `T27`, `T30`) |
| Stale metadata exports | `_data\annotations-audited\save\` | **Ignore entirely.** Never load; if pointed at a directory, exclude with the audit tool's `--exclude '*-save*.json'` |
| Eye-tracking gaze TSVs | `_data\eye-tracking-data-doc-space\` | 25 files (`p01.tsv`…`p31.tsv`, non-contiguous IDs, ~340 MB each, ~250 Hz Tobii export, tab-separated, UTF-8) |
| Document-space images | `_data\document-space-images\` | 36 PNGs named `T01.png` … `T30S.png`, matching metadata stems |
| Metadata audit tool | `_data\annotations-audited\audit_metadata.py` | **Reuse as the P2.6 tool. Invoke it (subprocess or import); do not rewrite it.** |

All pipeline outputs go to `data_processed/` (gitignored), keyed by a `data_version` tag from `configs/data.yaml`.

## 3. Repository skeleton (target layout)

Workspace root is the repo root. Layout per dev plan §1, plus `legacy/`:

```
├── configs/            # data.yaml, graph.yaml, encoder_selection.yaml, model_gnn.yaml,
│                       # model_transformer.yaml, train.yaml, splits.yaml, preprocessing.yaml, ablations/
├── schemas/            # segment.json, fixation.json, star_conditions.json (JSON Schema)
├── src/
│   ├── data/           # registry.py, gaze_load.py, segments.py, aoi_injection.py, coords.py,
│   │                   # gaze_assignment.py, fixations.py, loops.py, splits.py, validate.py
│   ├── text/           # encoder_selection.py, encoder.py            (Stage 2)
│   ├── graph/          # build.py, edges.py, diagnostics.py          (Stage 2)
│   ├── models/         # gnn.py, tokens.py, transformer.py, heads.py, biases.py (Stage 2)
│   ├── train/          # loop.py, losses.py, sampling.py             (Stage 2)
│   ├── eval/           # probes.py, loop_diagnostics.py, prototypes.py, metrics.py, viz/ (Stage 2)
│   └── utils/          # seeding, logging, run dirs, io.py (UTF-8 helpers), tracking.py
├── scripts/            # gaze_overlay_check.py, run_preprocessing.py, run_audit.py
├── tests/              # mirrors src/
├── fixtures/           # tiny synthetic trials for tests
├── legacy/             # gaze-feature-engineering.py (HMM porting reference, read-only)
├── reports/            # REPORT.md, DECISIONS.md, data_qc.md, gaze_checks/, metadata_audit/
├── artifacts/          # frozen v1 models (Stage 2)
├── data_processed/     # pipeline outputs (gitignored)
├── runs/               # experiment outputs (gitignored)
├── docs/               # the three binding specs (do not edit)
└── _data/              # raw inputs (READ-ONLY, gitignored)
```

Do **not** scaffold Stage 2 module contents during Stage 1 — create directories only when a milestone needs them.

---

# STAGE 1 — Repo skeleton + preprocessing pipeline

Ordered task list. Each task ends with green `pytest`, a `reports/REPORT.md` entry, and a commit.

## S1-T0 · M0 — Skeleton, schemas, fixtures

**Files:** repo layout above (Stage 1 dirs only); `schemas/segment.json`, `schemas/fixation.json`, `schemas/star_conditions.json`; `configs/data.yaml`, `configs/preprocessing.yaml` (placeholder values, commented); `src/utils/io.py` (UTF-8 read/write helpers all code must use); `fixtures/` with **two synthetic trials** (~10 segments, ~40 fixations each) exercising every edge type, empty-space fixations, and a multi-relation node pair (`SPATIAL_NEIGHBOUR` ∧ `SEMANTIC_CANDIDATE`); `tests/test_schemas.py`.

**Tasks:**
1. Preconditions PRE-2…PRE-4 (PRE-1 confirmed done by owner first).
2. JSON Schemas encoding the data contracts in dev plan §2.1–2.3 (segment, fixation, star-condition table).
3. Schema validators (`jsonschema`) wired to `src/utils/io.py`.
4. Synthetic fixture trials as JSON/parquet matching the schemas.

**Accept (dev plan M0):** `pytest` green on schema validators; both fixtures load through every schema.

## S1-T1 · P0 — Registries and identity

**Files:** `src/data/registry.py`, `tests/test_registry.py`; outputs `data_processed/{data_version}/registry/` (trial registry, document dimension registry, star-condition table parquet).

**Tasks:**
1. **Filename → identity parser** (one documented function): `T[n]` → non-eligible; `T[n]S` → (`T[n]`, `star_on`); `T[n]NS` → (`T[n]`, `star_off`). Applied to metadata files and document images; JSON-internal `question_id`/`trial_id` fields are ignored (empty by design). Loaders inject `trial_id`, `star_condition`, `question_id`, `question_type` from the registry.
2. **Trial registry** validation: exactly 30 trials; exactly 6 with S/NS variants (`T11, T12, T13, T21, T27, T30`); every metadata file has a matching document image and vice versa (36 each). **`question_type` (and `question_id`) come from the gaze TSVs' `Question type` column**, validated as constant per trial across all participants who viewed that trial (any disagreement → listed in `reports/data_qc.md` and stop); the registry stores the validated value per `trial_id` and is the sole source downstream.
3. **Document dimension registry:** read each of the 36 PNGs once with PIL → `{(trial_id, star_condition): (W_doc, H_doc)}` px. Sole source for P5 normalisation and the checker canvas.
4. **Star-condition assignment table** from the gaze TSVs' `Star Chart` column per (participant, trial). Validate: constant within episode; **exactly 3 star_on per participant among the 6 eligible trials**; zero star_on outside eligible trials. Conforms to `schemas/star_conditions.json`.
5. **Variant consistency check:** where both S and NS metadata exist, assert non-star content (aoi_annotations minus star chart, text_boxes, segments minus `star_concept`) identical. Report diffs.

**Accept:** unit tests on the parser (all three patterns + rejection case); registries built on real data; star-condition validations pass for all 25 participants (violations → listed in `reports/data_qc.md` and stop); variant-consistency report clean or triaged.

## S1-T2 · P1 — Gaze table pruning and tidying

**Files:** `src/data/gaze_load.py`, `tests/test_gaze_load.py`; outputs `data_processed/{data_version}/gaze_pruned/p{ID}.parquet` + per-episode QC table.

**Tasks:**
1. Load TSVs (UTF-8, tab-separated) with an explicit dtype map; filter to `Sensor == "Eye Tracker"`.
2. **Drop** exactly (preprocessing plan P1): `scroll_correction_flag`, gaze-direction vectors (6 cols), eye-position and gaze-point DACSmm (10) — **but see the ε-derivation note: extract the DACSmm medians needed for P6's ε before dropping**, per-eye MCSnorm gaze (4), `Project name`, `Ungrouped`, `Event`, `Event value`, `Trial Raw` (after cross-check vs `Trial`), `Sensor` (after filtering).
3. **Keep:** `Participant ID`, `Recording timestamp`, `Trial`, `Star Chart`, `Question type`, `Eye movement type`, `Eye movement type index`, `Gaze event duration`, `Validity left/right`, pupil columns (passthrough, never model inputs), `Fixation point X/Y` (+MCSnorm, HMM-continuity cross-checks only), `Gaze point X/Y`, **`Gaze point X/Y (doc)`**, `scroll_offset_y`, `scroll_ratio`, `gaze_region`, `AOI_label` + `AOI__*` one-hots (cross-checks only, never assignment), `correction_applied`, `left_offset_px`, `calibration_key`.
4. One documented snake_case rename map, in one place. Sort by (participant, trial, timestamp).
5. **QC per episode:** `correction_applied == False` counts (**resolved decision: trust these rows as-is** — doc coords with offset 0 are expected for non-scrolling episodes; count and report only); `Trial` vs `Trial Raw` disagreements; timestamp monotonicity; sample-period estimate (median diff); rows with empty `Trial` dropped and counted.

**Accept:** loader tests on a fixture TSV (drop/keep/rename verified column-by-column); all 25 participants pruned to parquet; QC table written into `reports/data_qc.md`.

## S1-T3 · P2 — Metadata compilation and audit

**Files:** `src/data/segments.py`, `scripts/run_audit.py` (thin wrapper invoking `_data\annotations-audited\audit_metadata.py`), `tests/test_segments.py`; outputs `data_processed/{data_version}/metadata/` (segment table, panel-region table per trial variant), `reports/metadata_audit/`.

**Tasks:**
1. **P2.6 audit first (hard gate for this stage):** run the existing `audit_metadata.py` against `_data\annotations-audited\complete` with `--image-dir _data\document-space-images --out reports/metadata_audit`. **Reuse the tool; do not rewrite it.** Non-zero exit (any ERROR) → stop; owner repairs metadata; re-run.
2. **Segment geometry:** bounding box = union of the segment's `box_ids` boxes from `text_boxes`; store union box + `n_boxes`, `n_lines`, per-box list. Validate: every `box_id` resolves; every box in ≤ 1 segment; unclaimed boxes reported.
3. **Canonical panel mapping** (config, not code — `configs/preprocessing.yaml`): `question→question`, `response→response`, `mark_scheme | mark_scheme_answers | mark_scheme_extra_information | level_descriptor → mark_scheme`, `commentary→commentary`, `star_chart→star_chart`, `general_ui/answer_scroll_bar/commentary_scroll_bar→ui`. Sub-AOI type retained as a segment feature. Second mapping for gaze-export AOI names (`AOI__Green_Answer_Box`/`AOI__Grey_Answer_Box`→response, `AOI__Advance`→ui, `AOI__Mark_Scheme`→mark_scheme, `AOI__Question`→question, `AOI__Response`→response, `AOI__Commentary`→commentary) for the Gate 1/Gate 2 cross-checks.
4. **Panel-region table** from `aoi_annotations` geometry (per trial variant) — used for empty-space categorisation, abstract panel nodes, checker outlines.
5. **Star handling:** `star_chart_annotations` is **legacy and ignored**. `star_concept` segments are self-sufficient (text from `corrected_text`, geometry via `box_ids`, identity from `star_id`). The star-chart AOI *region* stays required (P3 injection + panel priority).
6. **Field harmonisation** to `schemas/segment.json`: booleans → `bools` block; `ocr_confidence`, `aoi_manual`, `aoi_ambiguous` kept as QC fields; `corrected_text` is text-of-record (fallback `ocr_text` only if empty, flagged).
7. **Deterministic fallbacks (P2.7, documented in config comments):** `segment_role` derived from sub-AOI type where present (`answers`, `extra_information`, `level_descriptor`) else from `segment_type`; missing `aoi_id` resolved spatially (containing panel region, smaller region wins); duplicated `segment_order` within a panel tie-broken by geometry (top-to-bottom, then left-to-right) and flagged.
8. **`mark_point_id` policy (P2.8):** node metadata only, never an edge; audit reports coverage (already implemented in the tool).

**Accept:** audit exit code 0 on all 36 files (or every ERROR triaged by the owner and re-run clean); segment + panel tables for all 36 trial variants conform to `schemas/segment.json`; fallback applications counted and listed in `reports/REPORT.md`; unit tests for geometry union, panel mapping, and each fallback rule on fixtures.

## S1-T4 · P3 — AOI hit injection (star-chart + UI regions)

**Files:** `src/data/aoi_injection.py` (was star-only; generalised), `tests/test_aoi_injection.py`; outputs `data_processed/{data_version}/gaze_canonical/p{ID}.parquet` — the **AOI-injected gaze parquet, the canonical gaze table from P3 onward: written once; all downstream stages (P4–P7) read it, never the P1 pruned table directly** — plus per-episode injection QC.

**Tasks — sample level, before any event aggregation, in raw document px:**

1. **Star-chart injection (star_on episodes only; behaviour unchanged):** for each sample whose `(x_doc, y_doc)` falls strictly inside the star-chart AOI bbox, set `AOI__Star_Chart = 1`, `AOI_label = 'Star_Chart'`, and zero every other AOI one-hot (explicit override of the commentary mislabel). Star_off / non-eligible episodes untouched for this rule. Add `AOI__Star_Chart` to **all** episodes (constant 0 where inapplicable).

2. **UI-region injection (all episodes):** new hit columns `AOI__Answer_Scroll_Bar`, `AOI__Commentary_Scroll_Bar`, `AOI__General_UI` — set to 1 when the sample falls strictly inside the corresponding `aoi_annotations` region (`answer_scroll_bar`, `commentary_scroll_bar`, `general_ui`). Columns present on all episodes (constant 0 where the region is absent).

3. **Precedence:**
   - UI injections are **additive**: set `AOI_label` to the new region only when the sample has no existing content-AOI label (`NoAOI` / empty); **never** override a content hit (question / response / mark_scheme / commentary / star_chart).
   - Star-chart injection keeps its explicit commentary-override rule unchanged.
   - Where regions overlap, **smaller-region containment priority** applies (consistent with the existing panel-priority rule).

4. **QC:** per-episode hit counts for each new column + star-chart relabel counts / hit proportion. Note in QC that scrollbar regions are thin relative to gaze precision, so hit rates are indicative, not precise. Feeds Gate 1’s injection panels.

**Accept:** unit tests with synthetic samples straddling star and UI bboxes (inside/outside/boundary; UI never overrides content labels; star still overrides commentary; non-star episodes untouched by star rule); QC counts for all episodes (star_on: 75 episodes for star metrics; all ~750 for UI columns).

## S1-T5 · P4 — VISUAL GATE 1: metadata–gaze alignment (STOP POINT)

**Files:** `scripts/gaze_overlay_check.py` (Plotly, self-contained HTML, no server), `tests/test_overlay_smoke.py`; outputs `reports/gaze_checks/gate1/*.html`.

**Tasks — build the checker rendering, per (participant, trial), all in raw document px:**
1. Document image with segment bounding boxes (colour = canonical panel), panel-region outlines, star-chart boxes for star_on episodes; scroll-bar / general-UI region outlines.
2. Gaze overlaid two ways (toggle): raw sample scatter (density/alpha) and fixation points sized by duration, with a time slider/play control.
3. Each gaze point coloured by its **export AOI hit** (`AOI_label`, incl. injected `Star_Chart` and UI-region labels) so metadata boxes and AOI hits are visually cross-checkable — misalignment = colour spilling across box edges. Injected UI hits (`Answer_Scroll_Bar`, `Commentary_Scroll_Bar`, `General_UI`) rendered in their own distinct colours.
4. Star-injection panel for star_on episodes: relabelled samples highlighted, before/after counts from P3 QC; UI-injection summary with per-column hit counts (scrollbar rates flagged as indicative).
5. Summary stats: per-AOI-label counts; % gaze inside any segment box / any panel region / outside document.
6. **Batch mode** generating the stratified sample: every participant × ≥3 trials; star_on episodes covering all 6 eligible trials; every episode/file flagged by the P2 audit.

**Then STOP.** Owner reviews; findings triaged into metadata errors (fix JSON via audit report), identity/registry errors (fix P0), or coordinate problems (upstream). Sign-off recorded by the owner in `reports/DECISIONS.md`. **P5 does not start until sign-off. Do not self-certify.**

**Accept:** smoke test builds a report from fixtures with all panels non-empty; stratified batch generated for real data; owner sign-off entry exists in `reports/DECISIONS.md`.

## S1-T6 · P5 — Coordinate finalisation

**Files:** `src/data/coords.py`, `tests/test_coords.py`.

**Tasks:**
1. Working space = **raw document pixels** for all assignment, geometry, saccade computation, and checking.
2. **DOCnorm:** `x_docnorm = x_doc / W_doc`, `y_docnorm = y_doc / H_doc` from the P0 dimension registry; stored as extra columns; **never used for assignment**. Config switch `normalisation: docnorm | isotropic` (isotropic divides both axes by `W_doc`); default `docnorm` for HMM continuity, caveat documented in the config comment. The export's MCSnorm columns are screen-relative and scroll-unaware — never substitute them.
3. **Viewport features:** `y_screen = y_doc − scroll_offset_y`; normalised viewport document position = `scroll_offset_y / (H_doc − H_screen)`. Feed the P6 scroll feature set.

**Accept:** unit tests with hand-computed normalisation and viewport values; columns present on a real episode sample.

## S1-T7 · P6 — Fixation event construction and feature engineering

**Files:** `src/data/fixations.py`, `src/data/gaze_assignment.py`, `src/data/loops.py`, `tests/` for each; outputs one fixation parquet per episode under `data_processed/{data_version}/fixations/` conforming to `schemas/fixation.json`, plus per-episode QC rows.

**Tasks:**
1. **Event aggregation (ported from `legacy/gaze-feature-engineering.py`, core unchanged):** run-length event IDs over (`Eye movement type index`, `Eye movement type`); duration reconciliation (`dur_event_ms` vs timestamp span, mismatch flag, reconciled `dur_ms`); validity rates; pupil medians + missing rate (passthrough); `--min-valid-any` filter as config threshold. **Legacy-comparability check:** run old and new builders on the same input files; assert identical event segmentation and (up to renames) identical shared columns.
2. **Doc-space fixation position:** median of sample-level `(x_doc, y_doc)` within the event; DOCnorm of that median = model-facing position features; raw px retained for assignment and checking.
3. **Saccade geometry in document space:** `dx, dy, amplitude, angle, speed, is_regression` from first/last doc gaze points of saccade events, computed in raw px then expressed in DOCnorm; attached to the following fixation as `prev_sacc_*` (incl. `prev_sacc_found`), exactly as the legacy script does.
4. **Scroll features per fixation** from the `scroll_offset_y` trace: direction, displacement since previous fixation, instantaneous velocity, time since scroll onset/offset, during-scroll flag, normalised viewport document position, gaze-in-viewport y. **Input-only signals; never prediction targets.**
5. **Gaze→segment assignment policy** (`gaze_assignment.py`, config-driven, applied identically everywhere, raw document px):
   - Dilation margin ε per box. **ε derivation (resolved decision, self-contained):** recover mm-per-px by regressing paired DACSmm gaze columns against pixel gaze columns; viewing distance = median eye-position Z (DACSmm, ~700 mm); `ε_px = tan(0.5°) × distance_mm × px_per_mm`. Write derivation, inputs, and resulting value into `configs/preprocessing.yaml` as comments; report per-participant derived-distance spread as QC. **Fallback if regression unstable/implausible:** half the median vertical gap between adjacent text-box lines; record fallback in `reports/DECISIONS.md`. ε is a config default, not a scientific commitment.
   - Rules: (1) strictly inside exactly one box → that segment, confidence 1.0 interior decaying toward edge zone; (2) inside multiple boxes or within ε of ≥2 → smallest centre-weighted distance, `ambiguous=true`, runner-up `segment_id_alt` recorded, confidence reduced by best-vs-runner-up margin; (3) outside all boxes but within ε of one → nearest segment, confidence linear to 0 at ε; (4) beyond ε of all → empty-space category via P2 panel regions with **smaller-region containment priority** (star-chart over commentary; scroll-bar / general-UI regions over larger panels when contained). Empty-space categories are panel-specific content backgrounds **plus** the split UI categories `answer_scroll_bar` / `commentary_scroll_bar` / `ui_general` (P3-E1; not a single generic `ui` background), else `outside_document`.
   - Confidence = deterministic function of geometry, documented in the module docstring.
   - **ε sensitivity:** re-run at ×0.5 and ×1.5; report % of fixations whose assignment changes (feeds M8 sensitivity analysis).
6. **Visit/return + loop annotations** (`loops.py`, deterministic, config-driven, run at build time — single source for token features, attention biases, and the M7 D2 probe):
   - `visit_count`, `time_since_prev_visit`, `is_return`, `return_gap_events/ms`; `max_loop_gap` (default 20 events) splits short-loop returns from long-range revisits.
   - Template loops (panel-level `A→B→A` state machines, within event/ms window): response→mark_scheme→response; response→mark_scheme[level_descriptor]→response (panel refined by segment role); response→commentary→response; mark_scheme→response→mark_scheme; question→response→question; star variants (response→star_chart→response) in star_on episodes only. Overlapping loops all recorded.
   - Per-fixation outputs: `loop_role` (origin|pivot|closure|none), `loop_template_id` (multi-hot), `loop_origin_index`, gap features.
   - Per-template corpus frequency table; templates with < ~50 occurrences corpus-wide → dropped from D2 and recorded in `reports/DECISIONS.md`.
7. **Output:** per-episode fixation parquet (schema §2.2) tagged `data_version`; per-episode QC row (counts, empty-space %, edge-zone %, ambiguity %, mean confidence, correction/validity stats).

**Accept:** assignment-policy unit tests on synthetic geometry (interior, edge, overlap, multi-candidate, outside — hand-computed confidences); loop-detector unit tests on hand-constructed sequences (nested loops, overlapping templates, returns straddling `max_loop_gap`); legacy-comparability assertion green; ε derivation + sensitivity table generated; all ~750 episodes built; QC summary in `reports/data_qc.md`.

## S1-T8 · P7 — VISUAL GATE 2: assignment validation (= M1b, HARD STOP)

**Files:** extend `scripts/gaze_overlay_check.py` (same tool as Gate 1 — add layers, do not build a second codebase); outputs `reports/gaze_checks/gate2/*.html`.

**Tasks — Gate 1 canvas plus:**
1. Current fixation's **assigned segment highlighted**, with confidence, ambiguity flag, and scroll state in the info panel.
2. ε edge-zone warning rings; ambiguous fixations showing both candidate boxes; empty-space fixations rendered distinctly.
3. Per-episode summary: assignment/edge-zone/ambiguity/empty-space rates; distance-to-edge histogram; canonical-panel counts vs the export's `AOI__*` hits (the independent AOI columns and the new assignment must tell one consistent story).
4. Batch mode: same stratified sample as Gate 1, **plus every episode flagged by P5–P6 QC**.

**Then STOP.** Owner reviews and signs off in `reports/DECISIONS.md`. Systematic misalignment → fix upstream, rebuild, re-review. **No model code (M2 onward) before sign-off.** Stage 1 ends here.

**Accept:** smoke test on fixtures; stratified batch generated; owner sign-off in `reports/DECISIONS.md`; Stage 1 consolidation entry in `reports/REPORT.md`.

---

# STAGE 2 — Graph construction, model build, training preparation

Starts only after the P7 sign-off. Same discipline: one milestone at a time, tests first, REPORT.md entry per milestone. Full details live in the dev plan §3; this section is the map.

## S2-T1 · M2 — Text encoder selection and freeze (dev plan §4.4)

- `src/text/encoder_selection.py`: bake-off of 3–5 candidate sentence encoders (`all-mpnet-base-v2`, `e5-large-v2`, `bge-large-en-v1.5`, one small baseline) on a **manually reviewed pair set** — **owner input required: the pair set (student↔mark-scheme wording, related/unrelated response–criterion pairs, commentary paraphrases, command-word/level-descriptor cases) needs the owner's curation/review before the bake-off can be scored.** Selection metric fixed in `configs/encoder_selection.yaml`: ranking accuracy (related above unrelated, same anchor).
- Wrap winner as `TextEncoderV1` (`src/text/encoder.py`): frozen weights, documented preprocessing/pooling/dim/normalisation. Persist pair set to `artifacts/encoder_eval_pairs_v1.parquet`.
- **Accept:** bake-off table in `reports/`; `TextEncoderV1` card (model id + revision, pooling, dim, normalisation, thresholds); encoder hash pinned.

## S2-T2 · M3 — Automatic graph parser (dev plan §§5–6)

- `src/graph/build.py`: node features = concat[TextEncoderV1 embedding (L2-normalised, cached under `data_processed/embeddings/{graph_version}/`), categoricals (segment type, role, panel, level band, question type), booleans, formatting, normalised geometry, segment order]. Abstract panel nodes: learned type embedding + zeroed text slot.
- `src/graph/edges.py`, one function per edge type: `NEXT_SEGMENT`/`PREVIOUS_SEGMENT` (panel-grouped `segment_order`, geometry cross-check); `BELONGS_TO`; `SPATIAL_NEIGHBOUR` (n nearest within panel; attrs: distance, same-column, dx/dy); `SEMANTIC_CANDIDATE` (cross-panel only; allowed pairs from config; **response↔mark_scheme prioritised: k=3 + per-bullet coverage floor — every mark-scheme bullet gets ≥1 best response edge even below threshold, flagged `below_threshold`**; other pairs k=2 with threshold; attrs: cosine, rank, panel pair, below_threshold). `SAME_MARK_POINT` and `SAME_STAR` edges are **removed** — do not implement.
- **Star variants = per-variant construction** (DECISIONS.md M3-C1). Each S/NS graph is built from its own metadata/geometry. Regression = NS↔S node-correspondence on panel + normalised `corrected_text` + relative order within panel (geometry excluded). S-only allowlisted star-conditional segments (`configs/preprocessing.yaml` `star_conditional_text_patterns`) are flagged `is_star_conditional=true` and excluded from the correspondence requirement.
- `src/graph/diagnostics.py`: per-trial node/edge counts by type, degree distributions, similarity histograms, HTML/PNG per trial.
- **Accept:** per-edge-type unit tests incl. negative cases (same-panel semantic pairs excluded); all 36 graphs serialised under a `graph_version` tag as `data_processed/graphs/{graph_version}/{trial_id}__{star_condition}.pt`; NS↔S correspondence tests green for all eligible trials; diagnostics reviewed and summarised in `reports/`.

## S2-T3 · M4 — Compact GNN, standalone (dev plan §7)

- `src/models/gnn.py`: 2-layer edge-aware GAT (relation-type embeddings in attention, numeric edge features, residuals, edge dropout); outputs `x_v` and `h_v` separately. Throwaway sanity task: predict panel from `h_v` with panel features masked from `x_v`.
- **Accept:** shape/gradient tests; per-edge attention extractable; `x_v` vs `h_v` distinguishable (featureless-node panel probe); stable across 3 seeds.

## S2-T4 · M5 — Fixation tokens and dataset (dev plan §§8–9)

- `src/models/tokens.py` + dataset in `src/data/fixations.py`: token = concat[`x_v`, `h_v`, fixation features, prev-saccade, timing, scroll features (input-only), visit/return history, assignment confidence]. Empty-space fixations → learned panel-specific background embeddings (config switch: panel-specific | generic | drop).
- Episode `Dataset` with per-step targets: next panel; multi-hot next-relation (**all** relations between consecutive viewed nodes; `NO_DIRECT_RELATION` positive only when nothing else applies; empty-space transitions get their own label per config); candidate-ranking positives/negatives. Padding/collation.
- Loop annotations come from P6's `loops.py` output — already built; consumed here as features.
- **Accept:** dataset tests with hand-computed multi-hot vectors on fixtures; per-label relation frequency table on real data (feeds M6 class weights); throughput sanity check.

## S2-T5 · M6 — Causal loop-aware transformer + three losses (dev plan §§10–11)

- `src/models/transformer.py` (start: 4 layers, 4 heads, d_model 192, config-driven), `src/models/biases.py` (relative temporal bias; graph-relation bias per token pair; loop/return bias via `loop_origin_index`).
- `src/models/heads.py` + `src/train/losses.py`: **exactly three losses** — (1) next-panel softmax CE; (2) next-relation independent sigmoids + BCE with per-label weights from the M5 frequency table; (3) candidate next-node ranking, softmax-over-candidates CE, negatives = 8 easy + 4 hard (top-cosine unvisited) per positive, from config.
- `src/train/loop.py`: AdamW (lr 3e-4, wd 0.01, clip 1.0), early stopping on grouped-val loss, checkpointing, seeds {13, 42, 1337}, run-dir logging (metrics.jsonl + config snapshot + git hash). Equal loss weights (1,1,1) baseline. **Scroll-feature dropout p=0.3 per episode** (zero-mask). `src/utils/tracking.py`: MLflow local file store default, tags {milestone, ablation_id, fold, seed}; tracker failure never aborts a run.
- `src/data/splits.py` + `configs/splits.yaml`: **grouped 5-fold over participants = sole protocol for training, tuning, ablations.** Leave-one-question-out is post-hoc only, once, on the final frozen configuration.
- **Accept:** causal-mask leakage test (future perturbation → past outputs unchanged); overfit test on 2 fixture episodes (loss → ~0) — **this fixture-overfit test must pass identically on both the owner's laptop and the uni server before any full training run**; a pip lockfile (`pip freeze` or `pip-compile` output) committed alongside `environment.yml`, since the torch/PyG install line differs per CUDA setup and conda YAML alone underdetermines the env; full run across ≥3 seeds with stable curves; grouped-val metrics per loss (per-label AP for relations); V1+V2 visual report panels present.

## S2-T6 · M7 — Pre-registered diagnostic gate + temporal comparison (dev plan §11.4)

- `src/eval/loop_diagnostics.py` on **frozen** M6 embeddings, thresholds fixed in `configs/train.yaml` **before** looking at results: D1 return probe (embedding AUC ≥ feature-only AUC + 0.05); D2 loop-template probe vs within-episode label-shuffled baseline (macro-F1 margin from config); D3 true vs locally-shuffled subsequences.
- Gate failure → enable return/loop auxiliary losses (implemented behind config flags, default off), retrain, record in `reports/DECISIONS.md`.
- Primary temporal ablation: individual fixations vs merged segment visits. Fixed windows + bidirectional = secondary, config-flagged, only if time permits.
- **Accept:** diagnostic report with pass/fail per gate + documented decision; fixation-vs-visit comparison table; V2 stratifications + loop-diagnostic ROC panel.

## S2-T7 · M8 — Behaviour discovery, prototypes, validation (dev plan §12)

- `src/eval/prototypes.py`: GMM on (optionally PCA-reduced) train-fold embeddings; k by BIC over 4–12; stability via pairwise AMI across seeds/folds; soft memberships = posteriors; hard label only ≥ 0.6 posterior, else mixed/transitional.
- Prototype tracing: fingerprints (standardised mean differences over interpretable features, tornado chart), exemplars (≥5 participants, ≥5 trials each), document-space replay (`gaze_overlay_check.py --color-by prototype`), outcome anchoring (mixed models, participant random effect). **Owner input required: RTA data, observational reports, and per-trial outcome measures (confidence, difficulty, mental effort, time, marks) are not in `_data\` yet — owner supplies location/format before M8 validation runs.**
- `src/eval/probes.py`: trial-identity, participant-identity, question-type probes — measure and report, do not optimise away.
- Star-chart natural experiment (descriptive, episode-level, participant random effect).
- **HMM export hook (mandatory):** `runs/{run_id}/prototype_posteriors.parquet` keyed by (participant_id, trial_id, fixation_id, t_start_ms, duration_ms, seed, fold). **No HMM comparison code in Phase 1.**
- **Accept:** stability report; RTA alignment + reviewer judgements; outcome associations; probe accuracies; interpretation pack (HTML); posterior export; naming proposals drafted; V3+V4 panels.

## S2-T8 · M9 — Freeze the Phase 1 interface (dev plan 1G)

- Freeze into `artifacts/`: `graph_schema_v1`, `text_encoder_v1` (card + hash), `graph_encoder_v1.pt`, `behaviour_encoder_v1.pt`, `prototype_set_v1` (GMM params + naming + confidence threshold), soft pseudo-labels parquet.
- `load_phase1_interface()` single entry point; smoke test regenerating one fixture episode's embeddings bit-identically.
- **Accept:** all artefacts versioned with cards; round-trip test green; final consolidated `reports/REPORT.md`.

## S2-T9 · Pre-registered core ablations (after M6/M7, final config)

One config each under `configs/ablations/`, same seeds and grouped splits, compared via the V5 `compare_runs` report:
1. transformer without GNN (`x_v` only); 2. no loop-aware attention bias; 3. full model (reference); 4. no `SEMANTIC_CANDIDATE` edges; 5. no `SPATIAL_NEIGHBOUR` edges; 6. fixations vs merged segment visits.

**Exactly these six.** Secondary analyses stay behind config flags with justification in `reports/DECISIONS.md`.

## S2-T10 · Visualisation suite (`src/eval/viz/`, cross-cutting)

V1 training dynamics (from M6), V2 predictive performance (M6–M7), V3 embedding/behaviour maps (M7–M8), V4 episode interpretation (M7–M8), V5 cross-run comparison (after M6). Every run emits self-contained `runs/{run_id}/viz/report.html` from logged artefacts only; test-set panels behind a default-off flag. Smoke tests on fixture runs. Acceptance hooks amend M6/M7/M8/ablations as listed in dev plan §4b.

---

# Frozen scientific commitments (do not deviate in code)

Deviations, if ever needed, are proposed in `reports/DECISIONS.md` and approved by the owner — never silently implemented.

1. **Three initial losses only:** next-panel prediction; multi-label next-relation prediction; candidate next-node ranking. No masked reconstruction on the causal model; return/loop/contrastive losses only if a diagnostic gate fails (documented).
2. **Diagnostic gate D1–D3** with thresholds fixed in config before results are inspected (D1 default margin +0.05 AUC).
3. **Six core ablations** exactly as listed in S2-T9.
4. **Coverage-floor semantic edges:** response↔mark_scheme k=3 with per-bullet coverage floor (`below_threshold` flag); other allowed pairs k=2 + threshold; cross-panel only; allowed pairs: question↔response, response↔mark_scheme, response↔commentary, response↔star_chart, mark_scheme↔commentary.
5. **`SAME_MARK_POINT` and `SAME_STAR` edges removed**; `mark_point_id`/`star_id` are node metadata only.
6. **Grouped participant-held-out 5-fold CV = sole primary protocol**; leave-one-question-out post-hoc once on the final configuration; never tune on a held-out test fold.
7. **Loops via the deterministic detector** (single source for features, biases, D2 labels); templates < ~50 corpus occurrences dropped from D2 with a DECISIONS.md entry.
8. **Scroll features input-only** (never targets), with p=0.3 episode-level dropout.
9. **Visual gates P4 and P7 end with owner sign-off** recorded in `reports/DECISIONS.md`; no downstream work before sign-off (P5+ blocked by P4; all model code blocked by P7).
10. **HMM comparison deferred**; only the posterior export hook is built.
11. **Star variants: per-variant construction + verified non-star correspondence** (amended M3-C1, 2026-07-17). Each `(trial, star_condition)` graph is built from that variant’s own metadata/geometry. M3 regression = NS↔S correspondence on panel + normalised text + within-panel order (geometry excluded); S-only star-instruction segments are config-allowlisted (`is_star_conditional=true`), not required to match. **Supersedes** the earlier “base + overlay with identical non-star subgraph” wording.

# Resolved questions log (do not re-litigate)

| # | Question | Resolution |
|---|---|---|
| 1 | Document-space images location | `_data\document-space-images\` (36 PNGs, in-workspace, read-only) |
| 2 | `correction_applied == False` rows | Trust as-is (offset 0 expected for non-scrolling episodes); count and report per episode in P1 QC |
| 3 | HMM feature-engineering port source | `D:\Projects\EyeGaze\hierarchical-hmm-eye-tracking-analysis\data-processing\gaze-feature-engineering.py`, copied to `legacy/` as reference |
| 4 | Python environment | Conda env `gnn-gaze`, Python 3.11, CUDA PyTorch + PyG; `environment.yml` committed |
| 5 | Gaze edge tolerance ε | Derived self-contained from the data (DACSmm↔px regression for mm/px; median eye-Z for distance; ε = tan(0.5°)·d·px_per_mm); layout-based fallback via DECISIONS.md; ε is a config default, not a commitment |
| 6 | `T10-completee.json` typo | Owner renames to `T10-complete.json` before Stage 1 (precondition PRE-1) |
| 7 | Repo root | Workspace root; `git init` here; `_data/` gitignored |

# Known open items for the owner (not blocking Stage 1)

- **M2 pair set:** the manually reviewed encoder-evaluation pair set needs owner curation before the Stage 2 bake-off is scored.
- **M8 validation inputs:** RTA data, observational reports, and per-trial outcome measures (confidence, difficulty, mental effort, time on task, marks) are not yet in `_data\`; needed before M8's validation analyses.
- **Training runs:** the owner runs M6+ training themselves; the agent prepares code, configs, tests, and dry-run verification only.
