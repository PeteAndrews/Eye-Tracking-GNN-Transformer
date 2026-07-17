# Phase 1 Development Plan — Gaze-Informed GNN–Transformer for Examiner Behaviour Discovery

**Purpose of this document.** This is the implementation plan for Phase 1 only. It is written to be executed incrementally inside Cursor: work milestone by milestone, satisfy each milestone's acceptance criteria before moving on, and keep every scientific decision that has already been made (losses, ablations, diagnostic gates) fixed unless this document is explicitly revised. Phase 2 (GNN–LLM marking) is out of scope here; the only Phase 2 obligation in this codebase is that the artefacts frozen at the end (Graph Encoder v1, Behaviour Encoder v1, Prototype Set v1, Text Encoder v1) are versioned, loadable, and documented.

---

## 0. Working agreement for the coding agent (Cursor)

1. **One milestone at a time.** Do not scaffold future milestones speculatively. Each milestone ends with passing tests and a short `REPORT.md` entry in `reports/`.
2. **Config-driven everything.** No magic numbers in code. All hyperparameters, paths, thresholds, and switches live in YAML configs under `configs/`. Every run is reproducible from `config + seed + data version`.
3. **Data contracts are law.** Modules communicate through the schemas in §2. If a schema must change, update the schema file, its validator, and its tests in the same commit.
4. **Small model, strong discipline.** The effective independent units are 25 participants and 30 trial graphs. Prefer regularisation, multiple seeds, and grouped splits over model capacity. Never tune on a participant-held-out test fold.
5. **Pre-registered decisions are frozen.** The three initial losses (§6), the loop-diagnostic gate and thresholds (§8), and the six core ablations (§9) are scientific commitments. Do not add losses, ablations, or "quick experiments" beyond them without a documented decision in `reports/DECISIONS.md`.
6. **Determinism.** Seed `torch`, `numpy`, and `random` from config; log seeds and git commit hash into every run directory.
7. **Tests before training.** Every parser, dataset, and model component gets unit tests on synthetic fixtures before it ever touches real data.

---

## 1. Stack and repository layout

**Stack:** Python 3.11, PyTorch ≥ 2.2, PyTorch Geometric (GNN), `sentence-transformers` (text encoder candidates), `scikit-learn` (probes, clustering, metrics), `pandas`/`pyarrow` (tabular IO), `hydra-core` or plain YAML + `omegaconf` (configs), `pytest`, `matplotlib`/`plotly` (diagnostics), `mlflow` (local experiment tracking; W&B as a config-switchable alternative). Plain PyTorch training loop (no Lightning) to keep full control of masking, sampling, and logging; keep the loop small and tested.

```
examiner-behaviour/
├── configs/
│   ├── data.yaml               # paths, data version tags
│   ├── graph.yaml              # edge rules, k, thresholds
│   ├── encoder_selection.yaml  # candidate text encoders + eval pairs
│   ├── model_gnn.yaml
│   ├── model_transformer.yaml
│   ├── train.yaml              # losses, weights, optimiser, seeds
│   ├── splits.yaml             # grouped CV definitions
│   └── ablations/              # one YAML per pre-registered ablation
├── schemas/                    # JSON Schema files for all data contracts
├── src/
│   ├── data/
│   │   ├── validate.py         # M1: raw data validation & QC reports
│   │   ├── segments.py         # segment loading, canonical panel labels
│   │   ├── gaze_assignment.py  # M1b: fixation→segment assignment policy (edge tolerance, ambiguity, confidence)
│   │   ├── loops.py            # M5: loop event detector (returns, template loops) — feeds features, biases, D2 probe
│   │   ├── fixations.py        # fixation stream loading & feature build
│   │   └── splits.py           # grouped participant / question splits
│   ├── text/
│   │   ├── encoder_selection.py# M2: encoder bake-off
│   │   └── encoder.py          # frozen TextEncoderV1 wrapper
│   ├── graph/
│   │   ├── build.py            # M3: per-trial graph construction
│   │   ├── edges.py            # one function per edge type
│   │   └── diagnostics.py      # graph stats & visualisation
│   ├── models/
│   │   ├── gnn.py              # edge-aware GAT encoder
│   │   ├── tokens.py           # fixation token assembly / fusion
│   │   ├── transformer.py      # causal loop-aware transformer
│   │   ├── heads.py            # 3 loss heads
│   │   └── biases.py           # relation / temporal / loop attention biases
│   ├── train/
│   │   ├── loop.py             # training loop, checkpointing
│   │   ├── losses.py
│   │   └── sampling.py         # candidate negative sampling
│   ├── eval/
│   │   ├── probes.py           # trial-identity, question-type, loop probes
│   │   ├── loop_diagnostics.py # pre-registered gate (§8)
│   │   ├── prototypes.py       # clustering / soft prototypes + posterior export
│   │   ├── metrics.py
│   │   └── viz/
│   │       ├── training.py     # loss/metric curves, LR, grad norms
│   │       ├── performance.py  # confusion, PR, calibration, ranking plots
│   │       ├── embeddings.py   # UMAP/PCA maps, prototype overlays
│   │       └── episodes.py     # per-episode timelines & scanpath-on-graph
│   └── utils/                  # seeding, logging, run dirs, tracking.py (mlflow/wandb wrapper)
├── scripts/
│   └── gaze_overlay_check.py   # M1b: visual gaze-on-document sanity checker (pre-DL gate)
├── tests/                      # mirrors src/
├── fixtures/                   # tiny synthetic trials for tests
├── reports/                    # REPORT.md per milestone, DECISIONS.md
├── artifacts/                  # frozen v1 models (git-lfs or DVC)
└── runs/                       # experiment outputs (gitignored)
```

---

## 2. Data contracts

Define these as JSON Schema files in `schemas/` in M0 and validate all real data against them in M1. Field names below are the contract; adapt types to the actual export formats, but do the adaptation in loaders, never downstream. **The raw-input → contract pipeline (column pruning, document normalisation, metadata compilation, fixation building) is specified in the companion `gaze_preprocessing_plan.md` (stages P0–P7, with visual gates at P4 for metadata–gaze alignment and P7 for assignment), which supersedes/expands M1–M1b.** Notes from raw-data inspection: metadata `question_id`/`trial_id` fields are empty by design — identity is injected from filenames (`T21S` → trial T21, star_on); segment geometry is derived from `text_boxes` via `box_ids` (segments carry no coordinates); `aoi_annotations` supply panel regions; the star-condition table derives from the gaze export's `Star Chart` column; sample-level `Gaze point (doc)` columns confirm in-repo assignment from raw document-space coordinates is feasible.

### 2.1 Semantic segment (`schemas/segment.json`)
```jsonc
{
  "segment_id": "str, unique within trial",
  "trial_id": "str",
  "question_id": "str",
  "panel_label": "enum: question|response|mark_scheme|commentary|star_chart|ui",  // level descriptors are panel mark_scheme + segment_role level_descriptor (spatial rule)
  "corrected_text": "str",
  "segment_type": "str", "segment_role": "str",
  "level_band": "str|null",
  "mark_point_id": "str|null",
  "star_id": "str|null",
  "bools": { "command_word": true, "domain_term": false, "...": "see plan §4.3" },
  "formatting": { "bold": false, "italic": false, "formatted_prop": 0.0 },
  "geometry": { "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "n_boxes": 1, "n_lines": 1 },
  "segment_order": 4
}
```

### 2.2 Fixation event (`schemas/fixation.json`)
```jsonc
{
  "participant_id": "str", "trial_id": "str", "fixation_id": "str",
  "t_start_ms": 0, "duration_ms": 0,
  "segment_id": "str|null",                    // null → empty-space
  "empty_space_category": "enum|null",         // panel-specific bg or outside_document
  "panel_label": "enum",
  "assignment_confidence": 0.0,
  "scroll": { "direction": "enum: up|down|none", "displacement_px": 0.0, "velocity_px_s": 0.0,
               "t_since_scroll_onset_ms": 0, "t_since_scroll_offset_ms": 0, "during_scroll": false,
               "viewport_doc_position": 0.0, "gaze_viewport_y": 0.0 },
  "prev_saccade": { "amplitude": 0.0, "direction_deg": 0.0 }
}
```
Derived at load time (never stored redundantly upstream): `relative_trial_time`, `visit_count`, `time_since_prev_visit_ms`, `is_return`. **Note:** `segment_id`, `empty_space_category`, and `assignment_confidence` are produced (or re-derived and verified) by `data/gaze_assignment.py` under the explicit policy in M1b — they are not trusted blindly from upstream exports.

### 2.3 Trial graph (PyG `Data`, serialised per trial × star condition)
`x` (node features), `edge_index`, `edge_type` (int-coded relation), `edge_attr`, plus `node_id_lookup` (segment_id ↔ node index, including abstract panel nodes and empty-space token IDs), `graph_version` tag. **Star-chart conditions:** 6 trials are star-eligible, with presence randomised per participant (each participant sees exactly 3 with star charts on, varying which). Graphs are therefore keyed by `(trial_id, star_condition)`: star-eligible trials get two variants (`star_on` including star nodes; `star_off` without), non-eligible trials one. Files: `data/graphs/{graph_version}/{trial_id}__{star_condition}.pt`. A **condition assignment table** (`schemas/star_conditions.json`: participant_id × trial_id → star_condition) is a required input, validated in M1; the episode dataset joins through it to load the correct graph variant.

### 2.4 HMM reference — DEFERRED (do not implement in Phase 1)
The comparison against the two-level HMM is deferred until the HMM is rerun on document-space gaze with star-chart data included, so the comparison is like-for-like. Do not build `hmm_compare` functionality in this codebase yet. The only Phase 1 obligation is an export hook (see M8): per-fixation prototype posteriors saved in a tidy format keyed by (participant_id, trial_id, fixation_id, t_start_ms, duration_ms), so window-level aggregation against any future HMM output is possible without retraining.

---

## 3. Milestones

Each milestone below lists **scope → key implementation notes → acceptance criteria**. Map to plan phases: M1–M1b≈1A, M2–M3≈1B, M4≈1C, M5–M6≈1D, M7≈1E, M8≈1F, M9≈1G. M1b is a hard gate: no model code before its visual sign-off.

---

### M0 — Skeleton, schemas, fixtures
- Create repo layout, configs with placeholder values, JSON Schemas, and **two synthetic fixture trials** (≈10 segments, ≈40 fixations each) that exercise every edge type, empty-space fixations, and a multi-relation node pair (`SPATIAL_NEIGHBOUR` ∧ `SEMANTIC_CANDIDATE`).
- **Accept:** `pytest` green on schema validators; fixtures load through every schema.

### M1 — Data validation and canonical labels (plan 1A)
- `src/data/validate.py` produces `reports/data_qc.md`: unique-ID checks, geometry sanity, segment-order vs geometry consistency, Tobii-AOI ↔ canonical panel mapping table, **manual-label vs gaze-hit disagreement report** (flag segments where <50% of fixation hits agree with the manual panel), proportion of empty-space gaze overall and per panel, and **star-condition validation**: the participant × trial condition table loads, each participant has exactly 3 star_on episodes among the 6 eligible trials, and star-chart segments appear in the segment data only where the condition says they should.
- **Accept:** QC report generated on real data; every flagged disagreement triaged (fixed or documented) in `reports/REPORT.md`; empty-space proportion recorded.

### M1b — Gaze→segment assignment policy + visual sanity checker (pre-DL gate)
No model code (M4+) may start until this milestone's visual check has been manually approved.

**Assignment policy (`data/gaze_assignment.py`)** — one explicit, config-driven policy applied identically everywhere:
- All matching happens in **document space** (post scroll-correction), fixation point vs segment bounding boxes.
- **Edge tolerance:** each box is treated with a dilation margin ε (config, in document-space pixels; derive the default from tracker precision — ~0.5° visual angle at the study's viewing distance/scaling — and record the derivation in the config comment).
- **Assignment rules:** (1) point strictly inside exactly one box → that segment, confidence 1.0 in the interior, decaying towards the edge zone; (2) inside multiple overlapping boxes, or within ε of ≥2 boxes → assign to the box with smallest centre-weighted distance, set an `ambiguous=true` flag, record the runner-up `segment_id_alt`, and reduce confidence by the margin between best and runner-up; (3) outside all boxes but within ε of one → nearest segment, confidence decaying with distance-to-edge (linear to 0 at ε); (4) beyond ε of every box → panel-specific empty-space category (by containing panel region) or `outside_document`.
- **Confidence** is therefore a deterministic function of geometry (interior depth, distance-to-edge, best-vs-runner-up margin), documented in the module docstring; the schema field `assignment_confidence` is this value.
- **Sensitivity hook:** the policy re-runs cheaply with ε scaled ×0.5 and ×1.5; report the % of fixations whose assignment changes — this feeds the "sensitivity to gaze-assignment noise" analysis in M8 and quantifies how much of the data lives in the edge zone.

**Visual sanity checker (`scripts/gaze_overlay_check.py`)** — an interactive, human-in-the-loop check, not a unit test:
- Renders the document-space image (or reconstructed page) for a chosen (participant, trial) with all segment bounding boxes drawn and labelled by canonical panel colour.
- Overlays the fixation sequence with a **time slider / play control** (Plotly frames, self-contained HTML): current fixation as a marked point sized by duration, its **assigned segment's box highlighted**, the assignment confidence and ambiguity flag displayed, the current scroll state (during-scroll flag, velocity, viewport position) shown in the info panel, saccade line from the previous fixation, and empty-space fixations rendered distinctly.
- Edge-zone view: fixations within the ε band drawn with a warning ring so borderline assignments are visually obvious; ambiguous fixations show both candidate boxes.
- Summary panel per episode: % assigned / edge-zone / ambiguous / empty-space, distance-to-nearest-edge histogram, and per-panel assignment counts vs the Tobii AOI hits from M1 (the two QC views should tell one consistent story).
- Batch mode: generate the HTML for a stratified sample (every participant × ≥3 trials; **star-on episodes covering all 6 star-eligible trials**, since star-chart presence is randomised per participant; plus every episode flagged in M1's disagreement report) into `reports/gaze_checks/`.

**Accept:** policy unit tests on synthetic geometry fixtures (interior, edge, overlap, multi-candidate, outside cases with hand-computed expected confidence); ε-sensitivity table generated; the stratified visual sample manually reviewed and **signed off in `reports/DECISIONS.md`** (any systematic misalignment found → fix upstream mapping and re-run before proceeding); assignment-changed-% under ε-scaling recorded for M8.

### M2 — Text encoder selection and freeze (plan §4.4)
- `encoder_selection.py`: evaluate 3–5 candidate sentence encoders (e.g. `all-mpnet-base-v2`, `e5-large-v2`, `bge-large-en-v1.5`, one small baseline) on a **manually reviewed pair set** (student↔mark-scheme wording, related/unrelated response–criterion pairs, commentary paraphrases, command-word/level-descriptor cases). Selection metric fixed in config: ranking accuracy (related pair scored above unrelated pair, same anchor).
- Wrap winner as `TextEncoderV1` (frozen weights, documented preprocessing, pooling, dim, normalisation). Persist the reviewed pair set to `artifacts/encoder_eval_pairs_v1.parquet` — it is reused later as a Phase 2 reranker eval set.
- **Accept:** bake-off table in `reports/`; `TextEncoderV1` card written (model id + revision, pooling, dim, normalisation, thresholds); encoder hash pinned.

### M3 — Automatic graph parser (plan §§5–6)
- **Node feature assembly (`graph/build.py`):** each semantic node's feature vector `x` = concat[**`TextEncoderV1` embedding of `corrected_text`** (frozen, from M2, L2-normalised), one-hot/learned-embedding categorical features (segment type, role, canonical panel, level band, question type), boolean flags, formatting features, normalised geometry, segment order]. Abstract panel nodes get a learned type embedding + zeroed text slot. Embeddings are computed once per graph version and cached (`data/embeddings/{graph_version}/`) so encoder inference never runs inside the training loop.
- One function per edge type in `graph/edges.py`: `NEXT_SEGMENT`/`PREVIOUS_SEGMENT` (panel-grouped `segment_order`, geometry cross-check), `BELONGS_TO`, `SPATIAL_NEIGHBOUR` (n nearest within panel, edge attrs: distance, same-column, dx/dy), `SEMANTIC_CANDIDATE` (cross-panel only, allowed panel pairs from config, **panel-pair-aware**: response↔mark_scheme prioritised with k=3 plus a per-mark-scheme-bullet coverage floor — every bullet gets ≥1 best response edge even below threshold, flagged `below_threshold` — because bullet-vs-response matching is HMM-observed core behaviour and semantic edges are now the sole response↔mark-scheme channel; other pairs k=2 with threshold, nodes may receive none; edge attrs: cosine, rank, panel pair, below_threshold flag).
- **Star-condition variants are built as base + overlay, never as two independent builds:** `graph/build.py` constructs the base graph (all non-star content) once per trial, then `star_on` variants are produced by adding star-chart nodes and their edges (`BELONGS_TO`, spatial, semantic) on top of the frozen base. A regression test asserts that the non-star subgraph of `star_on` is identical (nodes, features, edges, attrs) to `star_off` for every eligible trial, so the variants cannot silently drift.
- `graph/diagnostics.py`: per-trial node/edge counts by type, degree distributions, semantic-edge similarity histograms, and an HTML/PNG visualisation per trial.
- **Accept:** unit tests per edge type on fixtures (including the negative cases: same-panel semantic pairs excluded); 30 real graphs built and serialised under a `graph_version` tag; diagnostics reviewed and summarised in `reports/`.

### M4 — Compact GNN, standalone (plan 1C, §7)
- `models/gnn.py`: 2-layer edge-aware GAT (PyG `GATv2Conv`-style with relation-type embeddings injected into attention, numeric edge features, residuals, edge dropout). Output preserves `x_v` (original) and `h_v` (contextualised) separately.
- Standalone sanity training: a throwaway node-level task (e.g. predict panel label from `h_v` with panel features masked out of `x_v`) purely to verify message passing, gradients, and attention-weight extraction. This model is discarded.
- **Accept:** shape/gradient tests; attention weights extractable per edge; `x_v` vs `h_v` remain distinguishable (probe: panel recoverable from `h_v` of a featureless node via neighbours); stable across 3 seeds.

### M5 — Fixation tokens and dataset (plan §§8–9)
- `models/tokens.py` + `data/fixations.py`: fixation token = concat[`x_v`, `h_v`, fixation features, prev-saccade features, timing, **scroll features** (direction, displacement, velocity, time since scroll onset/offset, during-scroll flag, viewport document position, gaze-in-viewport y — contextual signals of viewport movement, never prediction targets; gaze coordinates are already scroll-corrected upstream), visit/return history, assignment confidence]. Empty-space fixations map to learned panel-specific background embeddings (config switch: panel-specific | generic | drop — needed later for the secondary analyses).
- `torch.utils.data.Dataset` yielding one (participant, trial) episode = full token sequence + per-step targets (next panel; multi-hot next-relation vector; candidate-ranking positives/negatives). Padding/collation for variable lengths.
- **Loop event detector (`data/loops.py`) — the concrete representation of the loop construct.** A deterministic, config-driven pass over each fixation sequence, run at dataset-build time, producing per-fixation loop annotations that are the single source for features, attention biases, and the M7 D2 probe:
  - **Segment returns:** fixation on a segment previously fixated in the episode, with `gap_events` and `gap_ms` since the previous visit; config thresholds split short-loop returns (within `max_loop_gap` events, default 20) from long-range revisits.
  - **Template loops (panel level):** configurable patterns `A→B→A` over canonical panels completed within a window (events and/or ms), matching the semantic-checking construct — initial template set: response→mark_scheme→response, response→mark_scheme[level_descriptor]→response (panel refined by segment role — level descriptors live inside the mark scheme panel), response→commentary→response, mark_scheme→response→mark_scheme, question→response→question, plus star-chart variants for star_on episodes (response→star_chart→response). Templates are defined over panels, optionally refined by segment role. Detected via a small state machine per template; overlapping loops all recorded.
  - **Per-fixation outputs:** `is_return`, `return_gap_events/ms`, `loop_role` (origin | pivot | closure | none), `loop_template_id` (multi-hot — a fixation can close one loop and open another), `loop_origin_index` (token index of the loop's origin fixation).
  - **Consumers:** token features (M5) take `is_return`, gaps, `loop_role`, template multi-hot; the loop attention bias (M6) uses `loop_origin_index` to bias attention from closure toward origin tokens and toward same-segment predecessors; the **D2 loop-type probe (M7) uses `loop_template_id` as its labels** — this detector is therefore a hard dependency of the diagnostic gate; the deferred loop losses, if triggered, predict return/closure events from these same annotations.
  - **Accept additions:** detector unit tests on hand-constructed sequences (nested loops, overlapping templates, returns straddling `max_loop_gap`); per-template frequency table on real data (if a template has <~50 occurrences corpus-wide, drop it from D2 and record in DECISIONS.md).
- Multi-hot relation targets computed from the graph: **all** relations holding between consecutive viewed nodes; `NO_DIRECT_RELATION` positive only when no other label applies; transitions to/from empty-space handled by an explicit config rule (recommend: own label, excluded from `NO_DIRECT_RELATION`).
- **Accept:** dataset tests on fixtures verifying every target (hand-computed expected multi-hot vectors); per-label relation frequency table generated on real data (feeds the class-weighting decision in M6); throughput sanity check.

### M6 — Causal loop-aware transformer + three losses (plan 1D, §§10–11)
- `models/transformer.py`: small causal transformer (start: 4 layers, 4 heads, d_model 128–256, from config). Attention biases in `models/biases.py`:
  - relative temporal distance bias;
  - **graph-relation bias** between token pairs whose viewed nodes are connected (per-relation learned scalar);
  - **loop/return bias**: learned bias toward previous fixations on the same segment and toward tokens completing configured loop templates; return/loop *features* are already in the tokens (M5).
- `models/heads.py` + `train/losses.py`:
  1. next-panel: softmax CE;
  2. next-relation: independent sigmoids + BCE, **per-label weights** from M5 frequency table (config);
  3. candidate next-node ranking: scored by (behaviour state, candidate `x_v`+`h_v`, relation-to-current, visited flag, temporal features); softmax-over-candidates CE with sampled negatives — per positive, sample from same trial graph: n_easy random + n_hard (top cosine-similar unvisited) from config.
- `train/loop.py`: AdamW, cosine or plateau schedule, grad clip, early stopping on grouped-val loss, checkpointing, seed control, run-dir logging (metrics.jsonl + config snapshot + git hash). Equal loss weights as the committed baseline before any tuning. **Scroll-feature dropout:** with probability p_scroll_drop (config, default 0.3 per episode) all scroll features in an episode are zero-masked during training — Phase 2 agent trajectories have no viewport, so zeroed scroll must be in-distribution; report grouped-val metrics with scroll present vs masked as a standing diagnostic of scroll dependence. **Experiment tracking:** every run also logs to a tracker via a thin `utils/tracking.py` wrapper — backend from config (`mlflow` local file store by default, `wandb` optional, `none` for tests); log all scalar metrics per epoch, the flattened config as run params, tags for {milestone, ablation_id, fold, seed}, and the final `viz/report.html` as an artifact. The file-based run dir remains the source of truth; the tracker is for live monitoring and cross-run browsing only, so a tracker outage must never fail a run (wrap in try/except, warn once).
- Splits (`data/splits.py`, `configs/splits.yaml`): **primary and sole protocol for training, tuning, and all ablations = grouped 5-fold over participants.** Leave-one-question-out is **post-hoc only** (mirroring the HMM workflow): run once, on the final frozen configuration, after all tuning and ablations are complete — it involves retraining with each question held out, but is never part of the tuning loop and never repeated per ablation. Its results are reported as a descriptive robustness analysis (per plan §12.5's stance on question-type claims). Tune only within training folds.
- **Accept:** causal-mask leakage test (perturb future token → past outputs unchanged); overfit test on 2 fixture episodes (loss → ~0); full training run on real data across ≥3 seeds with stable curves; grouped-val metrics reported per loss (per-label AP for relations).

### M7 — Pre-registered loop-diagnostic gate + temporal comparison (plan 1E, §11.4)
- `eval/loop_diagnostics.py`, run on **frozen** M6 embeddings, thresholds fixed in `configs/train.yaml` before looking at results:
  - **D1 return probe:** logistic probe predicting return-to-current-segment within N events. Gate: probe AUC on embeddings must exceed AUC of a probe on raw token features alone by ≥ 0.05 (default; set final value before running).
  - **D2 loop-type probe:** multinomial probe over configured observable loop templates vs a within-episode label-shuffled baseline; gate: macro-F1 margin threshold from config.
  - **D3 local-order probe:** distinguish true vs locally-shuffled embedding subsequences.
  - If any gate fails → add return/loop auxiliary losses (implemented but disabled behind config flags) and retrain; record the decision in `reports/DECISIONS.md`. If prototypes in M8 are unstable/weakly separated → enable the contrastive objective, same procedure.
- Primary temporal-input ablation: individual fixations vs merged segment visits (visit-token aggregation per plan §8.2). Fixed windows and the bidirectional comparison are secondary — implement only if time permits, behind config flags.
- **Accept:** diagnostic report with pass/fail per gate and the resulting (documented) decision; fixation-vs-visit comparison table on grouped-val metrics.

### M8 — Behaviour discovery, prototypes, and validation (plan 1F, §12)
- `eval/prototypes.py`: fit soft prototypes on training-fold embeddings — **method committed here: Gaussian mixture on (optionally PCA-reduced) fixation embeddings, k selected by BIC over a config range (e.g. 4–12), stability-checked across seeds/folds (pairwise AMI of hard assignments ≥ threshold)**. Soft memberships = GMM posteriors; hard label only above a confidence threshold, else "mixed/transitional".
- **Prototype tracing workflow (`eval/prototypes.py` + `viz/`) — the defined path from cluster to behavioural meaning, four steps per prototype:**
  1. **Fingerprint (contrast statistics):** for each prototype vs all others, standardised mean differences over interpretable features only — panel/node-type occupancy, relation-type traversal rates, loop-template rates and `loop_role` mix, fixation duration, saccade amplitude, visit counts, relative trial-time position, assignment confidence — ranked and plotted as a tornado chart. This is automatic and model-free: it answers "what is statistically distinctive about this state".
  2. **Exemplars:** top-N contiguous subsequences by mean posterior, with enforced diversity (≥5 participants, ≥5 trials per prototype) so a prototype is never illustrated by one examiner's idiosyncrasy; each exemplar links to its episode timeline (V4).
  3. **Document-space replay:** the M1b overlay tool gains a `--color-by prototype` mode — fixations coloured by hard label (posterior as opacity) over the actual document image, so any prototype can be *watched* on real scripts. This is the primary artefact for RTA cross-referencing and reviewer judgement.
  4. **Outcome anchoring:** episode-level prototype proportions vs confidence, difficulty, RSME, time, mark, and cross-participant mark variance (mixed models, participant random effect) — connecting each state to the measures that matter for the paper's claims.
  The interpretation pack (below) is the assembled output of these four steps.
- `eval/probes.py`: trial-identity, **participant-identity** (scroll style and pacing are idiosyncratic, so participant leakage is plausible; same measure-and-report stance), and question-type probes on embeddings — reported as diagnostics/limitations, not optimised away (mitigation ladder per plan §12.5 only if egregious, via DECISIONS.md).
- **Primary validation (HMM comparison deferred — see §2.4):** prototypes are validated within Phase 1 against synchronised RTA and observational reports, plus the behavioural/outcome measures. Concretely: (a) RTA alignment — for a stratified sample of episodes with synchronised RTA, map RTA thematic codes onto the fixation timeline and report the prototype-occupancy distribution per RTA theme (contingency table + Cramér's V, with a permutation baseline shuffling prototype labels within episode); (b) manual review of the interpretation pack by at least one assessment-domain reviewer, with structured judgements (coherent / mixed / uninterpretable) per prototype; (c) association of episode-level prototype proportions with confidence, difficulty, mental effort, time on task, mark awarded, and cross-participant mark variance.
- **Star-chart natural experiment (descriptive, pre-registered):** because star-chart presence is randomised within the 6 eligible trials, compare behaviour on the *same question* between star_on and star_off participants — prototype-proportion differences, star-template loop rates, and next-panel transition patterns (episode-level, participant random effect, descriptive framing). This is the cleanest within-question behavioural contrast the design offers.
- **Statistical hygiene:** all prevalence/outcome analyses aggregate to episode level first (proportion of episode time per prototype), with participant as a random effect (mixed models via `statsmodels`); question-type comparisons reported descriptively only.
- Interpretation pack per prototype: representative sequences, panel/node-type occupancy, relations traversed, loop patterns, trial-phase position, association with the outcome measures — exported as an HTML report for manual review and RTA cross-referencing.
- **HMM export hook (mandatory, cheap):** save per-fixation prototype posteriors + hard labels to `runs/{run_id}/prototype_posteriors.parquet` keyed by (participant_id, trial_id, fixation_id, t_start_ms, duration_ms, seed, fold). This is the interface for the deferred HMM comparison; no comparison code is written now.
- **Accept:** prototype stability report; RTA-alignment analysis and reviewer judgements completed; outcome-measure associations reported; probe accuracies reported; interpretation pack generated; posterior export produced; descriptive naming proposals drafted for manual review.

### M9 — Freeze the Phase 1 interface (plan 1G)
- Freeze and version into `artifacts/`: `graph_schema_v1` (schemas + graph config snapshot), `text_encoder_v1` (card + hash), `graph_encoder_v1.pt`, `behaviour_encoder_v1.pt`, `prototype_set_v1` (GMM params + naming + confidence threshold), plus soft prototype pseudo-labels for the full development dataset (`parquet`).
- Loader round-trip: a single `load_phase1_interface()` entry point returning all frozen components; smoke test that regenerates embeddings for one fixture episode bit-identically.
- **Accept:** all artefacts versioned with cards (training config, data version, seeds, metrics); round-trip test green; final Phase 1 `reports/REPORT.md` consolidating M1–M8 results.

---

## 4. Pre-registered core ablations (run after M6/M7 on the final config)

One config each under `configs/ablations/`, each run over the same seeds and grouped splits:
1. transformer without GNN (tokens use `x_v` only);
2. full model without loop-aware attention bias;
3. full model (reference);
4. no `SEMANTIC_CANDIDATE` edges;
5. no `SPATIAL_NEIGHBOUR` edges (promoted from secondary; replaces the removed `SAME_MARK_POINT` ablation — that edge is dropped since `mark_point_id` labels individual mark-scheme bullets and is never shared between segments);
6. individual fixations vs merged segment visits (from M7).

Secondary analyses (empty-space variants, fixed windows, bidirectional comparison) stay behind config flags and run only if justified — record justification in `DECISIONS.md`.

---

## 4b. Visualisation suite (`src/eval/viz/`) — cross-cutting requirement

Every training run must produce a self-contained `runs/{run_id}/viz/report.html` (static Plotly, no server) assembled from the components below, so any run can be interpreted from its folder alone. All plots are generated from logged artefacts (`metrics.jsonl`, saved predictions, saved embeddings), never recomputed from the model, so reports can be regenerated offline. Split identity is visually explicit everywhere: **train / grouped-val / test curves and metrics always co-plotted**, test panels rendered only for final pre-registered evaluations (never during tuning — enforce with a config flag that defaults to off).

**V1 — Training dynamics (`viz/training.py`, available from M6):**
- per-loss curves (total + each of the three losses separately), train vs grouped-val, per seed with mean ± range band across seeds;
- learning rate schedule, gradient-norm trace, parameter-update norms;
- early-stopping marker and best-checkpoint epoch annotated on every curve;
- overfitting panel: train/val gap per loss over epochs;
- relation-loss detail: per-label BCE curves for the rare label (`SEMANTIC_CANDIDATE`) so class-weighting problems are visible during training, not after.

**V2 — Predictive performance (`viz/performance.py`, M6–M7):**
- next-panel: confusion matrix (row-normalised) per split; per-class F1 bar chart; calibration/reliability diagram + ECE;
- next-relation: per-label precision–recall curves and AP bar chart per split; co-occurrence heatmap of predicted vs true multi-hot patterns;
- candidate ranking: MRR and hits@{1,3,5} per split; rank histogram of the true next node; breakdown by relation-to-current-node and by visited/unvisited candidates;
- all of the above stratified by question type (descriptive, per §12.5) and plotted per fold to expose participant-fold variance;
- baseline overlays on every performance plot: majority-class / transition-frequency baseline and the feature-only (no-transformer) probe, so model lift is always visible against a floor.

**V3 — Embedding and behaviour-space maps (`viz/embeddings.py`, M7–M8):**
- 2-D UMAP and PCA projections of fixation embeddings (fit on train folds, transform val/test), coloured in linked panels by: prototype hard label, panel, question type, trial identity, participant, relative trial time, and return/first-visit — the trial-identity and question-type colourings double as the visual companion to the §12.5 probes;
- prototype cards: per-prototype occupancy over relative trial time, panel/node-type composition, top traversed relations, mean posterior confidence distribution;
- GMM diagnostics: BIC-vs-k curve with the selected k marked; cross-seed/fold stability heatmap (pairwise AMI);
- loop-diagnostic visuals for M7: probe ROC curves (embedding vs feature-only) with the pre-registered margin drawn as a reference line.

**V4 — Episode-level interpretation (`viz/episodes.py`, M7–M8):**
- episode timeline strip: fixation sequence coloured by prototype (soft posterior as opacity), with panel track, return events, and loop closures marked — the primary artefact for RTA cross-referencing in M8;
- scanpath-on-graph: the trial graph rendered (nodes positioned by document geometry) with the gaze trajectory overlaid and traversed edge types highlighted; selectable episodes;
- GNN attention view (from M4): per-trial graph with edge opacity ∝ attention weight, faceted by relation type;
- a curated default set per report: 2 highest-confidence, 2 most "mixed/transitional", and 2 randomly sampled episodes, plus any episode flagged during M1 QC.

**V5 — Cross-run comparison (`viz/` CLI, after M6):**
- a `compare_runs` entry point taking multiple run dirs and producing one report: ablation table with grouped-val metrics (mean ± sd across seeds/folds), paired per-fold difference plots against the reference model, and overlaid training curves — this is the artefact for §4's ablation write-up;
- tracker complement: because every run is tagged {milestone, ablation_id, fold, seed}, the full ablation matrix is also browsable live in the MLflow/W&B UI; `compare_runs` remains the canonical, citable output.

**Acceptance hooks (amend milestone criteria):** M6 additionally requires V1+V2 in the run report; M7 additionally requires V2 stratifications and the loop-diagnostic ROC panel; M8 additionally requires V3+V4; the §4 ablations require a V5 comparison report. All viz code gets smoke tests on fixture runs (report builds, all panels non-empty).

---

## 5. Default starting hyperparameters (all in configs, all provisional)

| Component | Default |
|---|---|
| Text encoder | winner of M2, frozen |
| Semantic edges | response↔mark_scheme: k=3 + per-bullet coverage floor; other pairs: k=2, threshold 0.5 (tune on train folds only) |
| Spatial neighbours | 4 per node, same panel preferred |
| GNN | 2 layers, hidden 128, 4 heads, edge dropout 0.2 |
| Transformer | 4 layers, d_model 192, 4 heads, dropout 0.2, context = full episode (cap ~2048 tokens; verify real episode lengths in M1) |
| Losses | equal weights (1,1,1) baseline |
| Negatives (ranking) | 8 easy + 4 hard per positive |
| Optimiser | AdamW, lr 3e-4, wd 0.01, grad clip 1.0 |
| Seeds | {13, 42, 1337} minimum |
| Gaze edge tolerance ε | ≈0.5° visual angle in document px (derive & document in config); sensitivity at ×0.5 / ×1.5 |
| Loop gate D1 margin | +0.05 AUC over feature-only probe |
| Scroll-feature dropout | p=0.3 per episode (zero-mask) |
| Prototype k range | 4–12 (BIC + stability) |
| Hard-label confidence | posterior ≥ 0.6 |

---

## 6. Definition of done for Phase 1

Phase 1 is complete when: all M1–M9 acceptance criteria pass; the six core ablations are run and tabulated; the loop-diagnostic gate decision is documented; prototypes are stable across seeds and validated through the RTA-alignment analysis, reviewer judgements, and outcome-measure associations; the per-fixation posterior export exists for the deferred HMM comparison; trial-identity and question-type probe results are reported as limitations; and the frozen v1 interface loads via `load_phase1_interface()` with a green round-trip test. Only then does Phase 2 development begin, against the frozen artefacts.

**Deferred (out of scope, tracked):** the convergent-validity comparison against the hierarchical HMM is deferred until the HMM is rerun on document-space gaze including star-chart trials. When that exists, the comparison consumes `prototype_posteriors.parquet` and the new HMM's window labels: aggregate prototype posteriors into the HMM's windows, compare assignment agreement (AMI/V-measure), transition structure (overlap-matched, per-row JSD), and macro-phase boundary F1 with a window-level tolerance. None of that requires retraining Phase 1 models.
