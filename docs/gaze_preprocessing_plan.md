# Gaze Data Preprocessing Plan — Phase 1 Input Pipeline

**Scope.** This plan covers everything between the raw exports and the model-ready fixation/segment tables consumed by the Phase 1 implementation plan (M3 onward). It supersedes and expands milestones M1–M1b of `phase1_dev_plan_cursor.md`. Ordering principle: **all raw-pixel-space work comes first** (registries, pruning, metadata, star injection, and an early visual alignment gate), and only then normalisation and feature engineering — so metadata–gaze linkage is verified visually before anything is derived from it. Two visual gates: Gate 1 (P4) checks metadata–gaze *alignment* early; Gate 2 (P7, = M1b) checks fixation→segment *assignment* late. The pre-DL rule stands: **no model code until P7 is signed off.**

**Inputs (as inspected):**
- **Sample-level gaze TSVs** (~250 Hz Tobii export, one per participant–trial or per participant): fixation/saccade classification already present (`Eye movement type`, `Eye movement type index`), screen and MCSnorm coordinates, **document-space gaze** (`Gaze point X/Y (doc)`, scroll correction already applied and validated upstream), `scroll_offset_y`, `Star Chart` condition flag, AOI hit columns, validity, pupil.
- **Metadata JSON per trial variant** (e.g. `T21S-complete-update.json`): `aoi_annotations` (panel/UI regions), `text_boxes` (line-level geometry), `segments` (semantic segments with rich boolean/role metadata, geometry **only via `box_ids`**), `star_chart_annotations`. `question_id`/`trial_id` fields are empty by design — identity comes from the filename.
- **Document-space images** per trial variant: source of document dimensions (and background for the visual checker).
- Existing feature-engineering script (`gaze-feature-engineering.py`) from the HMM analysis: its event-aggregation core is ported, then extended.

---

## P0 — Registries and identity

1. **Filename → identity rule.** One documented parser with three patterns: `T[n]` → non-eligible trial (no star chart possible); `T[n]S` → `trial_id=T[n]`, `star_condition=star_on`; `T[n]NS` → `trial_id=T[n]`, `star_condition=star_off` (eligible, off). Applied to metadata files and document images; validated against the trial registry (exactly 6 trials with S/NS variants). Loaders inject `trial_id`, `star_condition`, and (from the trial registry) `question_id`/`question_type` into every record; the empty JSON fields are ignored, not trusted.
2. **Document dimension registry.** Read each document-space image once (PIL); store `{trial_id, star_condition} → (W_doc, H_doc)` px. These dimensions are the sole source for document normalisation (P2) and the checker canvas (Gates 1 and 2).
3. **Star-condition assignment table.** Derived from the `Star Chart` column in the gaze data per (participant, trial). Validation: constant within episode; exactly 3 star_on per participant among the 6 eligible trials; zero star_on outside eligible trials. This table is the input required by the implementation plan (§2.3).
4. **Variant consistency check.** Where both `T21` and `T21S` metadata exist, assert the non-star content (aoi_annotations minus star chart, text_boxes, segments minus `star_concept`) is identical — the metadata-level twin of the M3 base+overlay rule.

## P1 — Gaze table pruning and tidying

**Drop:** `scroll_correction_flag` (legacy diagnostic), gaze-direction vectors (6 cols), eye-position and gaze-point DACSmm (10), per-eye MCSnorm gaze (4), `Project name`, `Ungrouped`, `Event`, `Event value`, `Trial Raw` (after cross-checking against `Trial`), `Sensor` (after row filtering).

**Keep:** `Participant ID`, `Recording timestamp`, `Trial`, `Star Chart`, `Question type`, `Eye movement type`, `Eye movement type index`, `Gaze event duration`, `Validity left/right`, pupil columns (passthrough for analyses; **not** model inputs in this project), `Fixation point X/Y` (+MCSnorm, for HMM-continuity cross-checks only), `Gaze point X/Y`, **`Gaze point X/Y (doc)`**, `scroll_offset_y`, `scroll_ratio`, `gaze_region`, `AOI_label` + `AOI__*` one-hots (cross-checks only, not assignment), `correction_applied`, `left_offset_px`, `calibration_key`.

**Tidy:** filter to `Sensor == "Eye Tracker"`; snake_case rename map (documented, one place); dtype coercion; row-order sort by (participant, trial, timestamp) exactly as in the original script.

**QC:** rows where `correction_applied == False` counted and reported per episode; `Trial` vs `Trial Raw` disagreements; timestamp monotonicity; sample-period estimate per file (median diff, as in the original script).

## P2 — Metadata compilation and audit (per trial variant)

1. **Segment geometry from text boxes.** Segments carry no coordinates; each segment's bounding box = union of its `box_ids` boxes from `text_boxes` (store the union box plus `n_boxes`, `n_lines`, per-box list for multi-line segments). Validation: every `box_id` resolves; every box belongs to ≤1 segment; unclaimed boxes reported.
2. **Canonical panel mapping and the sub-AOI design.** The gaze-facing AOIs are deliberately coarse (e.g. `Mark_Scheme`), while metadata `aoi_type`s may be finer **sub-AOIs** that add node-level granularity without overriding the overarching panel: `mark_scheme_answers`, `mark_scheme_extra_information`, and `level_descriptor` all collapse to canonical panel `mark_scheme`, and a generic `mark_scheme` template region (used in trials without sub-AOIs) maps identically — so panel-level behaviour (next-panel loss, empty-space categories, Gate 1 cross-checks) is comparable across trials regardless of annotation granularity, while the sub-type is retained as a segment feature. Full table (config, not code): `question→question`, `response→response`, `mark_scheme | mark_scheme_answers | mark_scheme_extra_information | level_descriptor → mark_scheme`, `commentary→commentary`, `star_chart→star_chart`, `general_ui/answer_scroll_bar/commentary_scroll_bar→ui`. A second mapping covers the gaze export's AOI names (`AOI__Green_Answer_Box`, `AOI__Grey_Answer_Box` → response; `AOI__Advance` → ui; …) for the Gate 1/Gate 2 cross-checks.
3. **Panel regions.** `aoi_annotations` geometry becomes the panel-region table: used for panel-specific empty-space categorisation, abstract panel nodes, and the checker's panel outlines.
4. **Star handling.** The `star_chart_annotations` array is **legacy and ignored** — it is not used anywhere in the pipeline. `star_concept` segments are self-sufficient: text from `corrected_text`, geometry from their `box_ids` like any other segment, identity from `star_id`. The star-chart **AOI region** (in `aoi_annotations`) is unrelated to the legacy array and remains required for the P3 injection and panel priority.
5. **Field harmonisation to the implementation-plan schema (§2.1):** booleans map to the `bools` block; `ocr_confidence`, `aoi_manual`, `aoi_ambiguous` retained as QC fields; `corrected_text` is the text-of-record (fall back to `ocr_text` only if empty, flagged).
6. **Per-file graph-readiness audit (hard check before graph building).** For every trial-variant metadata file, assert: `level_band` filled on all `level_descriptor` segments; `star_id` filled on all `star_concept` segments; all `box_ids` and `aoi_id`s resolve; `corrected_text` non-empty; no duplicate (`aoi_id`, `segment_order`) pairs. Failures are listed per file for manual repair (implemented in `audit_metadata.py`: per-file PASS/FAIL console summary, a CSV of issues with entity ids and suggested fixes, and a grouped markdown report; ERROR = breaks graph construction, WARN = documented fallback applies, INFO = worth a look; non-zero exit on any ERROR so it can gate the pipeline) — the T21S example passes almost everything but exhibits each near-miss exactly once (one star segment missing `star_id`, one empty `aoi_id`, one duplicated order pair, one empty text), so this audit is not hypothetical.
7. **Deterministic fallback rules (documented, applied only where the audit tolerates them):** `segment_role` is unfilled in the data, so it is **derived from the sub-AOI type** where present (`answers`, `extra_information`, `level_descriptor`) and from `segment_type` otherwise; a missing `aoi_id` is resolved spatially — the panel region containing the segment's box union, with containment priority to the smaller region; star text comes from the segment's `corrected_text`; duplicated `segment_order` within a panel is tie-broken by geometry (top-to-bottom, then left-to-right) and flagged.
8. **`mark_point_id` policy.** `mark_point_id` labels each bullet point within the mark scheme — it is node metadata (bullet identity, kept for interpretation and Phase 2 traceability), **not** a linking identifier, and the formerly planned `SAME_MARK_POINT` edge is removed accordingly. The audit reports per-file coverage (filled/total mark-scheme bullets) as an INFO-level completeness statistic; a shared id across two segments is now itself a WARN (likely annotation error).

## P3 — Star-chart AOI hit injection (sample level, star_on episodes only)

The original Tobii AOI processing treated the star chart as an extension of the commentary region, so sample-level `AOI_label`/`AOI__*` hits inside the star chart read as `Commentary`. This stage corrects that, at **sample level, before any event aggregation**, using the raw (unnormalised) document-space gaze coordinates:

1. applies only to `star_on` episodes (from the P0 condition table); star_off and non-eligible episodes are untouched;
2. for each sample whose `(x_doc, y_doc)` falls inside the star-chart AOI bounding box (from `aoi_annotations`, raw px, strict containment to mirror how the original AOI hits were computed), set `AOI__Star_Chart = 1`, set `AOI_label = 'Star_Chart'`, and zero `AOI__Commentary` (and any other one-hot) for that sample — an explicit override of the commentary mislabel;
3. the `AOI__Star_Chart` column is added to the one-hot family for **all** episodes (constant 0 where not applicable) so downstream schemas are uniform;
4. QC: per star_on episode, report the count of samples relabelled commentary→star_chart and the star-chart hit proportion; rendered directly in Gate 1's star-injection panel.

This correction primarily serves (a) the planned HMM rerun on document-space data with star charts — it is the enabling step for that rerun — and (b) the Gate 1/Gate 2 cross-checks between the export's AOI hits and the new segment assignment, which would otherwise disagree by construction inside the star chart. The GNN pipeline's own segment assignment (P6.5) never uses AOI labels, but its **panel-region priority rule must match**: where the star-chart region overlaps the commentary region, containment priority goes to the smaller (star-chart) region, for empty-space categorisation and spatial `aoi_id` fallback alike.

## P4 — Visual Gate 1: metadata–gaze alignment (early, raw pixel space)

**Purpose:** observe, before any normalisation or feature engineering, that the correct metadata is linked to the eye-gaze coordinates — and that the P3 star-chart injection is working. Everything here runs in **unnormalised document pixels**, which is why P0–P4 come first and normalisation waits until P5.

**Inputs (deliberately minimal):** the document-space image; the pruned gaze table (sample-level doc coordinates and/or classified fixation points); the AOI hit columns (including the injected `AOI__Star_Chart`); and the compiled metadata (segment boxes, panel regions, star annotations).

**The checker renders, per (participant, trial):**
- the document image with segment bounding boxes (colour = canonical panel), panel-region outlines, and star-chart boxes for star_on episodes;
- gaze overlaid two ways (toggle): raw sample scatter (density/alpha) and fixation points sized by duration, with a time slider/play control;
- each rendered gaze point coloured by its **export AOI hit** (`AOI_label` including `Star_Chart`), so metadata boxes and AOI hits can be visually cross-checked against each other — misalignment shows up as colour spilling across box edges;
- a star-injection panel for star_on episodes: samples relabelled commentary→star_chart highlighted, with the before/after counts from P3 QC;
- summary stats: per-AOI-label counts, % of gaze inside any segment box / any panel region / outside document.

**Review protocol:** stratified sample (every participant × ≥3 trials; star_on episodes covering all 6 eligible trials; anything flagged by the P2 audit). Findings triaged into: metadata errors (fix the JSON, guided by the audit report), identity/registry errors (fix P0), or coordinate problems (upstream). Sign-off recorded in `DECISIONS.md`. **P5 onward does not start until this gate passes.** Note this gate validates alignment, not assignment — fixation→segment assignment with confidence/ambiguity does not exist yet and gets its own gate at P7.

## P5 — Coordinate finalisation (document normalisation)

1. **Working space = raw document pixels.** All segment assignment, geometry, saccade computation, and visual checking run in `(x_doc, y_doc)` px. This matches the metadata geometry (also px) with no further transform.
2. **Document normalisation ("DOCnorm", the MCSnorm analogue).** `x_docnorm = x_doc / W_doc`, `y_docnorm = y_doc / H_doc` using the P0 dimension registry. This mirrors Tobii's MCSnorm convention (position ÷ media size) and preserves continuity with the HMM feature space, but computed against the full scrolled document rather than the screen. Stored as additional columns; **never** used for assignment. The export's existing MCSnorm columns cannot be substituted — they are screen-media-relative and unaware of scroll.
   - *Config switch:* `normalisation: docnorm | isotropic` — isotropic divides both axes by `W_doc`, keeping vertical distances comparable across trials of different document lengths. Default `docnorm` for HMM continuity; the caveat (per-trial `H_doc` makes y-distances trial-relative) is documented.
3. **Feature-space convention.** Fixation positions (P6.2) and saccade geometry (P6.3) are expressed in **DOCnorm** units as model features, mirroring the HMM's MCSnorm-based feature space; the underlying computation and all assignment run in raw document pixels first, then normalise. Document normalisation is therefore a required preprocessing step ahead of feature engineering, not an optional extra.
4. **Viewport features preserved:** `y_screen = y_doc − scroll_offset_y` recovers gaze-in-viewport position; `scroll_offset_y / (H_doc − H_screen)` gives normalised viewport document position. These feed the scroll feature set in P6.

## P6 — Fixation event construction and feature engineering

Port the HMM script's core unchanged, then extend. Per (participant, trial):

1. **Event aggregation (ported):** run-length event ids over (`Eye movement type index`, `Eye movement type`); duration reconciliation (`dur_event_ms` vs timestamp span, mismatch flag, reconciled `dur_ms`); validity rates (`valid_any_rate`, `valid_both_rate`); pupil medians + missing rate (passthrough); `--min-valid-any` filter retained as a config threshold.
2. **Document-space fixation position (new):** median of sample-level `(x_doc, y_doc)` within the fixation event (there is no fixation-level doc column in the export; the median mirrors the script's MCSnorm treatment). The **DOCnorm coordinates computed from this median are the model-facing fixation position features**; raw px retained for assignment and visual checks.
3. **Saccade geometry in document space (changed):** `dx, dy, amplitude, angle, speed, is_regression` computed from first/last doc gaze points of saccade events and **expressed in DOCnorm units** (computation in raw px, then normalised) — not screen MCSnorm as in the original, since screen-space geometry is distorted whenever scrolling moves content mid-saccade. Attach to the following fixation as `prev_sacc_*` exactly as the original does (including `prev_sacc_found`).
4. **Scroll features per fixation (new, per plan §8.1):** from the `scroll_offset_y` trace within/around the event — direction, displacement since previous fixation, instantaneous velocity, time since scroll onset/offset, during-scroll flag, normalised viewport document position, gaze-in-viewport y. Input-only signals; never targets.
5. **Segment assignment (M1b policy, unchanged):** fixation doc point vs segment boxes with dilation margin ε, ambiguity handling, runner-up recording, geometry-derived `assignment_confidence`; fixations beyond ε of all segments → panel-specific empty-space category via the P2 panel regions (star-chart region taking containment priority over commentary), else `outside_document`. ε-sensitivity re-runs at ×0.5/×1.5 with assignment-change % reported.
6. **Visit/return + loop annotations (new):** `visit_count`, `time_since_prev_visit`, `is_return`, then the loop event detector (implementation plan M5/§10.1) producing `loop_role`, `loop_template_id` (multi-hot), `loop_origin_index`, gap features. Star templates evaluated only in star_on episodes.
7. **Output:** one fixation table per episode (parquet), conforming to the implementation-plan fixation schema (§2.2), tagged with `data_version`, plus a per-episode QC row (counts, empty-space %, edge-zone %, ambiguity %, mean confidence, correction/validity stats).

**Legacy-comparability note:** the ported aggregation is verified by running old and new builders on the same input files and asserting identical event segmentation and (up to renames) identical shared columns — protecting comparability with the HMM feature space before the extensions diverge.

## P7 — Visual Gate 2: assignment validation (= M1b, the hard gate)

As specified in the implementation plan M1b, run **after** P6 so assignments exist to inspect. This is the same tool as Gate 1 extended with assignment-specific views (build Gate 1 first; Gate 2 adds layers rather than a second codebase): document image canvas with all segment boxes (colour = canonical panel) and panel-region outlines; fixation replay with time control; current fixation's assigned segment highlighted with confidence, ambiguity flag, and scroll state in the info panel; ε edge-zone warning rings; ambiguous fixations showing both candidate boxes; empty-space fixations rendered distinctly; per-episode summary (assignment/edge-zone/ambiguity/empty-space rates, distance-to-edge histogram, canonical-panel counts vs the export's `AOI__*` hits — the independent AOI columns and the new segment assignment must tell one consistent story).

Stratified sample as Gate 1, plus every episode flagged by P5–P6 QC. Manual review and sign-off recorded in `DECISIONS.md`; systematic misalignment → fix upstream, rebuild, re-review. **Model development begins only after sign-off.**

---

## Order of operations and outputs

`P0 registries → P1 prune/tidy → P2 metadata compile (+ audit) → P3 star AOI injection → P4 VISUAL GATE 1 (alignment, raw px) → P5 normalisation → P6 fixation build (assignment, scroll, loops) → P7 VISUAL GATE 2 (assignment)`

Deliverables: condition assignment table; document dimension registry; compiled segment/panel/star tables per (trial, star_condition); per-episode fixation parquet + QC report; visual check HTMLs; sign-off entry. All stages config-driven, deterministic, and versioned under a single `data_version` tag consumed by graph building (M3) and everything downstream.
