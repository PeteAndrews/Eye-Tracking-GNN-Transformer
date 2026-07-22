# Decisions log

Gate sign-offs (owner), fallback activations, and documented deviations from the
pre-registered plan. Sign-offs are written by the owner, not the coding agent.

---

## M3-C1 — Amend frozen commitment #11: per-variant graphs + correspondence (2026-07-17)

**Owner decision.** Replace “base + overlay with identical non-star subgraph” with:

> **Per-variant construction + verified non-star correspondence** (text / panel /
> relative order identity; geometry per-variant; allowlisted star-conditional extras).

**Rationale.** Gaze→segment assignment depends on correct per-variant geometry.
Star and non-star UIs differ slightly, so there is no single shared geometric base.
Each `(trial_id, star_condition)` graph is built from that variant’s own
metadata/geometry.

**Replacement regression test (M3).** For each eligible trial, build an NS↔S
node-correspondence table matched on canonical panel + normalised `corrected_text`
+ relative order within panel. Assert every NS non-star segment maps 1:1 to an S
segment with identical text and panel. Geometry is excluded from the comparison.
Segments present in S only (e.g. star-instruction commentary) must appear on the
config allowlist (`configs/preprocessing.yaml` → `star_conditional_text_patterns`),
be flagged `is_star_conditional=true`, and are excluded from the correspondence
requirement.

**Amends:** PLAN.md frozen commitment #11; `.cursorrules` star-variant bullet;
PLAN.md S2-T2 / M3 acceptance text.

---

## P0-V1 — S/NS metadata variant consistency (superseded in part, 2026-07-17)

**Original finding (morning).** Non-star content was not byte-identical across
S/NS (geometry drift + segment asymmetries, especially T30).

**Owner fix.** Replaced NS document-space PNGs; redrew leftover S→NS AOI boxes
against the correct NS canvas; corrected metadata.

**Re-check after fix (afternoon).**

| Trial | Correspondence (panel+text+order) | Notes |
|---|---|---|
| T12, T13, T21, T27, T30 | **PASS** | T30 asymmetry **gone** (was annotation artefact). Soft geometry/id drift only. |
| T11 | **FAIL** | Still has commentary segmentation differences + empty `corrected_text` on NS `ann_segment_060` (P2.6 audit ERROR). |

**P0/P2 status.** Dimension registry rebuilt. Audit clean except **T11NS**
`no_text` on `ann_segment_060`. Correspondence check uses M3-C1 rules; T11 still
blocks a fully clean P0/P2 until that segment is fixed and commentary texts align
(or are explicitly allowlisted if star-conditional).

**Path note (updated).** Audited JSONs flattened back to
`_data/annotations-audited/complete/`. `audit_metadata.py` remains at
`_data/annotations-audited-legacy/audit_metadata.py`. `T10-completee.json`
typo may still be present — rename when convenient.

---

## P3-E1 — Generalise P3 to AOI hit injection (2026-07-17)

**Amendment.** P3 scope extends from star-chart-only injection to **AOI hit
injection**, without changing star-chart behaviour.

**Additions (all episodes):**
- Columns `AOI__Answer_Scroll_Bar`, `AOI__Commentary_Scroll_Bar`, `AOI__General_UI`
  set when sample `(x_doc, y_doc)` is strictly inside the matching
  `aoi_annotations` region (`answer_scroll_bar`, `commentary_scroll_bar`,
  `general_ui`). Present on all episodes (0 if region absent).

**Precedence:**
- UI injections are **additive**: update `AOI_label` only when there is no
  existing content-AOI label (NoAOI/empty); never override content hits.
- Star-chart injection retains its commentary-override rule.
- Overlaps: smaller-region containment priority (same as panel-priority rule).

**P6 knock-on:** empty-space categories split UI into `answer_scroll_bar` /
`commentary_scroll_bar` / `ui_general` (not a single generic ui background).
`schemas/fixation.json` enum updated.

**QC / Gate 1:** per-episode hit counts for each new column; distinct colours in
Gate 1; note that scrollbar regions are thin vs gaze precision — rates are
indicative, not precise.

**No change** to star-chart injection semantics.

---

## P4 — Visual Gate 1 sign-off (2026-07-17)

Reviewed stratified Gate 1 sample (75 episodes). Alignment acceptable.
Signed off for P5. — Peter Andrews

---

## P7 — Visual Gate 2 sign-off (2026-07-20)

Reviewed Gate 2 sample (stratified + P6 QC flags). Assignment acceptable.
Signed off for Stage 2. — Peter Andrews

---

## M2-A1 — Hard + easy negatives in encoder pair set (2026-07-20)

**Owner amendment.** Draft encoder-eval triples use a ~50/50 mix of
`hard_within_trial` (unrelated from the same trial) and `easy_cross_trial`
(unrelated from another trial). Bake-off reports ranking accuracy overall and
by `negative_type`; hard-negative accuracy is the tie-breaker. Promote accepts
same-trial unrelated and does not enforce the draft hard/easy mix after review.

---

## M2-A2 — Retire command_word / level_descriptor; sampler fixes (2026-07-20)

**Owner amendment after batch-1 review (13/48 kept).**

1. Drop `command_word` — question instructions have no valid related criterion.
2. Drop `level_descriptor` as anchor/related — near-identical boilerplate across
   trials (duplicate-detection, not response↔criterion matching).
3. Sampler fixes: never `anchor==related`; never cross-trial related; exclude
   rubric-instruction and level-descriptor segments from the **related** slot
   (they may still be unrelated); prefer Jaccard-matched content MS bullets;
   diversify relateds (avoid repeated fallback segments).

Batch-2: ~15 new `response_mark_scheme` + `commentary_paraphrase` candidates
appended to keepers with `reviewed=false`.

---

## M2 — Encoder bake-off complete (2026-07-20)

**Pair set:** 25 reviewed triples (14 hard / 11 easy) promoted to
`artifacts/encoder_eval_pairs_v1.parquet`.

**Bake-off (ranking accuracy):**

| id | overall | hard | easy |
|---|---|---|---|
| mpnet | 0.84 | 0.786 | 0.909 |
| bge_large | 0.84 | 0.786 | 0.909 |
| e5_large | 0.80 | 0.714 | 0.909 |
| mini_baseline | 0.76 | 0.714 | 0.818 |

**Initial freeze:** `mpnet` by config-order tie-break (overall+hard tied with bge).

**Backend note:** bake-off uses `transformers` mean-pooling rather than
`sentence-transformers` 5.6 (torchcodec/FFmpeg breakage on this machine). Same HF
checkpoints and pooling convention. `HF_HUB_DISABLE_XET=1` required on this Windows
host (hf_xet native fetch SIGILL).

---

## M2-A3 — Owner override: TextEncoderV1 = bge_large (2026-07-20)

**Owner decision.** Re-freeze TextEncoderV1 as `bge_large` (`BAAI/bge-large-en-v1.5`,
dim=1024, revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`).

**Rationale.** mpnet and bge tied on overall (0.84) and hard (0.786); config-order
tie-break is arbitrary at this sample size. bge is stronger on
`response_mark_scheme` (0.82 vs 0.73) — the sub-task that maps to
`SEMANTIC_CANDIDATE` edges and Phase 2 retrieval — and is the stronger public
retrieval encoder. **This is a tie broken on the highest-stakes sub-score, not a
decisive bake-off win.**

**Propagated:** `configs/graph.yaml` → `text_embedding_dim: 1024`,
`graph_version: g1_bge1024`; card at `artifacts/text_encoder_v1_card.json`.

---

## M6-PRE — Pre-M6 locks: semantic pairs, relation weights, three-loss scope (2026-07-20)

### 1) SEMANTIC_CANDIDATE allowed panel pairs (read-only on `g1_bge1024`)

Diagnostic: `scripts/diag_semantic_panel_pairs.py` →
`reports/graph_diagnostics/g1_bge1024/semantic_panel_pairs.md`.

| panel pair | directed edges | graphs ≥1 |
|---|---:|---:|
| question ↔ response | 636 | 36/36 |
| response ↔ mark_scheme | 1624 | 36/36 |
| response ↔ commentary | 1051 | 32/36 |
| response ↔ star_chart | 312 | 6/36 |
| mark_scheme ↔ commentary | 1530 | 32/36 |

**Verdict.** All five `graph.yaml` pairs are represented. None near-empty when both
panels exist. Gaps explained: commentary pairs absent on 4 tiny trials with no
commentary segments (T01/T02/T19/T25); `response↔star_chart` only on the 6
`star_on` graphs (star segments absent elsewhere). No unexpected pairs outside
the allowlist. **No rebuild.**

### 2) Relation BCE weights (M6-W1)

**Decision.** Use the frozen M5 corpus table
`reports/relation_frequencies_m5.json` directly — **do not recompute per fold/split**.
Scheme: inverse-frequency (`n_trans / count`), then **clip at `clip_max`**.
Resolved clipped weights are **computed at load time** from the M5 JSON +
`clip_max` via `src/train/relation_weights.py`.

**Amendment (2026-07-22).** Ceiling locked at **`clip_max: 12.0`** (not 8.0 / 10.0).
Rationale: inverse-frequency chosen over sqrt-inverse; clip at 8.0 was rejected
because it collapsed four of six active labels onto the same weight mass.
At 12.0, PREVIOUS (raw 13.3) and SEMANTIC (raw 25.9) clip; NEXT (9.92) and
EMPTY (9.54) stay below the ceiling and remain differentiated.

**Zero-count labels.** `exclude_zero_count: true`. `BELONGS_TO` (0 consecutive-
fixation transitions) is **excluded** from the BCE head/target set — not given
weight 0 on a dead sigmoid. Active labels (6): NEXT, PREVIOUS, SPATIAL,
SEMANTIC, NO_DIRECT, EMPTY_SPACE. Recorded in `configs/train.yaml`.

### 3) Three-loss scope (frozen)

M6 trains **exactly** three losses, equal objective weights (1,1,1):
1. next-panel softmax CE
2. multi-label next-relation BCE (per-label clipped weights above)
3. candidate next-node ranking (softmax over candidates; 8 easy + 4 hard)

Return / loop auxiliary and contrastive objectives remain behind
`losses.return_aux` / `loop_aux` / `contrastive` with `enabled: false` until an
M7 diagnostic gate failure + DECISIONS entry. Confirmed in `configs/train.yaml`.

---

### 4) Windows CPU access violation workaround (M6-W2) (2026-07-20)

**Symptom.** `run_m6_train.py` dies with `Windows fatal exception: access violation`
(exit `0xC0000005`) after ~400–500 training steps — no Python traceback. Stack
frames jump around (backward, Linear, even numpy in `__getitem__`), consistent
with heap corruption.

**Context.** This host runs `torch 2.6.0+cu126` with `cuda.is_available() == False`
(CUDA wheel, no visible GPU).

**Mitigations locked in config/code:**
- Lazy episode loading + shared ≤36 graph cache (earlier OOM fix).
- `max_seq_len: 256`; `biases.graph_relation.enabled: false`.
- Empty-space emb applied in `BehaviourModel`, not Dataset `__getitem__`.
**Also.** Cursor/IDE shells may set `CUDA_VISIBLE_DEVICES=-1`, which hides a
working GPU from PyTorch even when `nvidia-smi` shows the card. `run_m6_train.py`
now clears that sentinel before importing torch and accepts `--device cuda`.

**Also (2026-07-22).** Stale `checkpoint_last.pt` with `global_step=555` from B=1
chunked training caused a silent hang when restarting with `--batch-size 8`
(only ~75 batches/epoch): the loop skipped every batch with no progress prints.
Fix: mid-epoch step skip only in `--max-steps` chunked mode; full-epoch runs
restore weights but start at batch 0. Add `--fresh` to ignore last checkpoint.
Checkpoints now store `batch_size`.

**Also (2026-07-22, M6-W4).** Windows + CUDA torch: any `torch.cuda.*` call
*before* the first pandas/pyarrow parquet read hard-crashes with `0xC0000005`
on smoke/`__getitem__` (reproduced: CUDA API → then `pd.read_parquet` → AV;
parquet-first → then CUDA → OK). `run_m6_train.py` warms parquet via
`src/utils/arrow_cuda.py` before any CUDA device call, and defers GPU init
until after the smoke item. Dataset reads go through `read_parquet`.

---

### 5) ReduceLROnPlateau (M6-W3) (2026-07-22)

**Decision.** Add `torch.optim.lr_scheduler.ReduceLROnPlateau` on grouped-val
`loss_total`, config under `optim.scheduler` in `configs/train.yaml`
(default: factor 0.5, patience 3, min_lr 1e-6). LR is logged in `metrics.jsonl`
and restored from `checkpoint_last.pt`. Scheduler patience is kept below
`early_stopping.patience` (10) so LR can decay before stop. Disable with
`optim.scheduler.name: none`.

---

### 6) Diagnostic run profile → full-matrix schedule (M6-W5) (2026-07-22)

**Diagnostic.** `runs/m6/fold0_seed13` (B=8, lr 3e-4, clip_max 12, fold 0, seed 13):
best grouped-val total **2.4496 @ epoch 64**; ReduceLROnPlateau stepped LR at
epochs **47 / 62 / 68 / 72**; stopped by ES patience 10 at epoch **74** with
train/val gap **0.62**. Per-label go/no-go on `checkpoint_best.pt`:
SEMANTIC_CANDIDATE AP **0.165** vs base-rate **0.037** → **GO** (see
`reports/m6_fold0_seed13_predictive_metrics.md`).

**Schedule lock for 15-run matrix.** `max_epochs: 100` (ceiling only — ES fires
well before); `early_stopping.patience: 10` retained. Train/val gap noted as a
known property; **no extra regularisation** before the matrix — revisit only if
per-fold results look unstable.

**Tracking.** `tracking.backend: mlflow` (local file store) with tags
`{milestone, ablation_id, fold, seed}`; `ablation_id: baseline` for this matrix.

---

### 7) M7 diagnostic gate on fold0 seed13 best ckpt (M7-G1) (2026-07-22)

**Status.** The numbers in this subsection are the *first* diagnostic run and are
**superseded** by §7c (corrected labels). Kept for audit trail only.

**Run.** Frozen embeddings from `runs/m6/fold0_seed13/checkpoint_best.pt`
(grouped-val participants P03/P06/P10/P12/P21). Report:
`reports/m7_fold0_seed13_diagnostics.md` (buggy D2 re-detect).

| Gate | Result | Detail |
|---|---|---|
| D1 return probe | **FAIL** | emb AUC 0.700 · feat 0.680 · margin **0.021** < 0.05 |
| D2 loop template | PASS† | macro-F1 0.707 vs shuffled 0.275 · margin 0.432 |
| D3 subsequence | PASS | AUC 0.631 |

†D2 active set was only **four** templates; LD and star were falsely zeroed — see §7b.

**D1 failure shape (not a null result).** Embeddings *did* outperform the
feature-only baseline (0.700 vs 0.680) but fell short of the pre-committed
**+0.05** AUC margin. The feature-only probe is strong because return-related
fields (`is_return`, `visit_count`, `time_since_prev_visit` / gap features) are
already model *inputs*, so the baseline has near-direct access to return
history. **Threshold not changed** — remedy applied as pre-registered.

**Decision (scoped to the failed gate only — owner 2026-07-22; confirmed on
corrected labels in §7c).** The pre-registered rule adds the loss for the
construct whose gate failed. Corrected D2 still passes, so loop-template
structure is already encoded without supervision; adding `LoopRoleHead` would
inflate the objective set for no measured benefit.

- `losses.return_aux.enabled: true`, weight **0.5** — `ReturnHead`,
  return-within-horizon BCE, H=20.
- `losses.loop_aux.enabled: false` (and contrastive off).
- Tracking: `ablation_id: baseline_m7_return_aux`, milestone M7.

**Overnight matrix** must use this return-aux-only config (`--fresh`). Do not
treat the pre-aux diagnostic checkpoint as the matrix reference.

### 7b) Loop-detector bug scope (M7-G1b) (2026-07-22)

**Verdict: diagnostic re-detect only — training annotations are clean.
No M5 rebuild / no `data_version` bump.**

| Path | What happened |
|---|---|
| **P6 → episode parquet → training** | `loop_role`, `loop_template_id`, `loop_origin_index` already written. `EpisodeDataset` uses those columns when present and does **not** re-annotate. Model training therefore saw correct loop features / attention bias origins (including star and level-descriptor). |
| **M7 D2 first re-detect** | Rebuilt templates from `panel_id` alone → stripped `segment_role` → LD template forced to 0 and absorbed into plain `response→mark_scheme→response`. |
| **M7 D2 star still 0 after parquet reload** | `collate_episodes` omitted `star_condition`, so every episode defaulted to `not_eligible`, star templates were skipped, and star-on parquet paths missed. Fixed: collate now passes `star_condition`. |

Corpus-wide behavioural counts (full-length P6 `annotate_loops`, not truncated):
level-descriptor template **~2310**; star template **~1001** across 75 `star_on`
episodes. The star figure is a real behavioural signal for M8 star-chart
analysis — the first diagnostic’s 0 was an artefact, not absence.

Fold 0 train∪val = 750 episodes = the full corpus, so a correct re-detect on
that set *must* find star loops; finding 0 proved the diagnostic path (not the
shared training detector) was broken.

### 7c) Corrected M7 gate re-run (M7-G1c) (2026-07-22)

**Fixes applied before re-run:** D2 attach reloads full parquet rows (role);
`collate_episodes` includes `star_condition`. Shared M5/P6 detector unchanged.

**Report:** `reports/m7_fold0_seed13_diagnostics_corrected.md` (also under
`runs/m6/fold0_seed13/m7_diagnostics_corrected/`). Same frozen pre-aux ckpt.

| Gate | Result | Detail |
|---|---|---|
| D1 return probe | **FAIL** | emb AUC **0.682** · feat **0.680** · margin **0.003** < 0.05 |
| D2 loop template | **PASS** | macro-F1 **0.533** vs shuffled ~0.214 · margin **0.318** (harder 6-class) |
| D3 subsequence | **PASS** | AUC **0.872** |

**Corrected D2 template counts** (fold0 train∪val, truncated to `max_seq_len`):

| Template | Count |
|---|---|
| `response→mark_scheme→response` | 16909 |
| `mark_scheme→response→mark_scheme` | 11665 |
| `question→response→question` | 3250 |
| `response→commentary→response` | 1620 |
| `response→mark_scheme_level_descriptor→response` | 907 |
| `response→star_chart→response` | **346** (full-length corpus ≈ **1001**) |

All six templates active (none dropped <50). D2 still clears the +0.05
macro-F1 margin cleanly under the harder label set.

**Auxiliary-loss config (from corrected gates).** Unchanged from §7:
`return_aux` on (0.5); `loop_aux` off. D1 still fails the margin (now even
tighter vs feature-only); D2/D3 still pass → no loop_aux activation.

**Supersession.** Original M7-G1 gate metrics (§7 / `m7_fold0_seed13_diagnostics.md`)
are **not** the decision basis. Corrected §7c is authoritative for gate
outcomes; the return-aux-only remedy still stands.

**Next (superseded by M6-W6).** Do **not** lock `return_aux` or launch the
matrix until fold0/seed13 is retrained at `max_seq_len=1536` and M7 is
re-gated. Truncation may have been the D1 failure mode.

---

### 8) Full-sequence lift — truncation validity (M6-W6) (2026-07-22)

**Problem.** `max_seq_len: 256` (M6-W2) discarded **40.2%** of all fixations
and truncated **43.1%** of episodes. Marking-phase structure lives in episode
tails; relative trial-time features and long-range returns/loops were
systematically cut. Full table: `reports/truncation_analysis_m6w6.md`.

| Overall | |
|---|---|
| Mean / median / max length | 292 / 207 / **1520** |
| % episodes > 256 | 43.1% |
| % fixations discarded @ 256 | **40.2%** |

| By condition | median | % eps >256 | % fix lost |
|---|---:|---:|---:|
| `star_on` (n=75) | 526 | 89.3% | 56.8% |
| `star_off` (n=75) | 526 | 89.3% | 55.8% |
| `not_eligible` (n=600) | 155 | 31.5% | 29.8% |

Star/LD decomposition (annotated template **rows**):

| | % abs index ≥ 256 | % rel `t/T` ≥ 0.5 |
|---|---:|---:|
| Star (n=1283) | 64.2% | **74.8%** |
| Level-descriptor (n=2929) | 59.1% | 58.7% |

`star_on` mean length **580** vs other **260**. Both “star_on episodes are
long” and “star loops are late in relative time” are true.

**What M6-W2 actually constrained.** Not a 16 GB CUDA OOM. Host was running
the CUDA torch wheel **without a visible GPU** (`cuda.is_available()==False`);
training AVed (`0xC0000005`) after ~400–500 CPU steps. Dense
`biases.graph_relation` (T×T×R `pair_relations`) was a co-factor and remains
**disabled**. The `chunked` flag in checkpoints is only `--max-steps` **resume
chunking**, not a long-sequence alternative path.

**Memory profile (RTX 3080 Ti Laptop 16 GB, graph_relation OFF, fwd+bwd):**
`reports/mem_profile_seqlen.json`

| T × B | Peak alloc |
|---|---|
| 256 × 8 | 0.33 GB |
| 512 × 8 | 0.68 GB |
| 1024 × 8 | 1.62 GB |
| **1536 × 8** | **2.85 GB** |

**1536 covers 100% of episodes** (max 1520). Constraint lifts on this CUDA
host at batch 8 with large headroom. Plan budget was ~2048 with M1
verification; M1 found max ~1520 — the scientific budget was already correct.

**Decision.**
- `configs/model_transformer.yaml`: `max_seq_len: 1536`.
- `losses.return_aux.enabled: false` until the full-length M7 gate is re-run
  (do not lock an aux remedy measured under truncation).
- Tracking tag: `ablation_id: baseline_fullseq_1536`.
- Retrain fold0/seed13 `--fresh` under `runs/m6_fullseq/` (keeps the old
  `runs/m6/fold0_seed13` truncated-regime artefacts intact), then M7 +
  predictive; only then decide aux and matrix.

**In progress → superseded by M6-W7.** Bias-off full-seq train under
`runs/m6_fullseq/` was aborted mid-epoch 9; left intact as **reference only**.
Ship baseline is full-seq **with** graph-relation bias (next section).

**Supersession.** All prior fold0/seed13 results measured at cap 256 are
**truncated-regime artefacts** and are superseded, including: 74-epoch
convergence (best val 2.45 @ ep 64), SEMANTIC AP 0.165, ranking MRR ~0.789,
and both M7 gate reports (original + corrected labels). They remain on disk
for audit only.

**Lesson.** An engineering workaround (M6-W2 AV mitigation) became a
methodological constraint without a scientific cost review. Future host-local
mitigations that change the data the model sees must get an explicit
validity check before results are treated as binding.

---

### 9) Graph-relation bias restored (M6-W7) (2026-07-22)

**Context.** Research plan §10 specifies three attention biases (temporal,
graph-relation, loop/return). `graph_relation` was disabled under M6-W2 as a
Win-CPU AV workaround and never re-enabled after CUDA training became
available — a second engineering workaround that became an architectural
constraint (same class of mistake as the 256 cap).

With the bias off, graph structure reaches the transformer only via node
embeddings inside each token; there is no path for attention to prefer tokens
whose nodes share a SEMANTIC/SPATIAL/… edge. That plausibly explains the
ranking tie vs the feature-only cosine probe and is directly relevant to M7.

**Pre-flight (RTX 3080 Ti Laptop 16 GB).**

Memory with graph bias ON (`reports/mem_profile_seqlen_graphbias.json`):

| T × B | Peak alloc |
|---|---|
| 1536 × 8 | **3.20 GB** |
| 1536 × 4 | 1.62 GB |

Batch **8** remains safe.

Correctness with bias ON on CUDA (`reports/graphbias_long_t_checks.json`):
all four checks **PASS** at T=768 and T=1536 — causal leak (B=1 and B=2
padded), loss padding invariance, padded-batch NaN/Inf+grad.

**Decision.**
- `biases.graph_relation.enabled: true`
- `max_seq_len: 1536`; `return_aux` / `loop_aux` / contrastive **off** until
  M7 on this architecture
- Tracking: `ablation_id: baseline_fullseq_1536_graphbias`
- New run root: `runs/m6_fullseq_graphbias/` (do not overwrite
  `runs/m6_fullseq/fold0_seed13`)
- Standing guard: every run’s `run_meta.json` / `train_summary.json` records
  truncated episode + fixation counts (`src/train/truncation_stats.py`)

**Supersession.** Metrics from bias-off runs (including aborted
`baseline_fullseq_1536`) are **reference-only**, not the Phase-1 baseline.

---

### 10) Full-seq + graph-bias gate → return_aux (M7-G2) (2026-07-22)

**Run.** `runs/m6_fullseq_graphbias/fold0_seed13` (tag
`baseline_fullseq_1536_graphbias`; cap 1536; graph_relation on; no aux).
Reports: `reports/m7_fold0_seed13_diagnostics_fullseq_graphbias.md`,
train summary under the same stem.

**Ranking (paper keep).** Model MRR **0.8405** vs feature-only cosine **0.7958**;
ahead at every cut (hits@1 0.750 vs 0.728; @3 0.916 vs 0.820; @5 0.962 vs
0.881). Two runs ago these tied — full sequences + restored graph-relation
bias are the first direct evidence the learned representation adds value over
static features.

**D1 margin trajectory (same +0.05 threshold; H=20 throughout):**

| Measurement | emb AUC | feat AUC | margin | Notes |
|---|---:|---:|---:|---|
| Truncated + buggy D2 re-detect | 0.700 | 0.680 | 0.021 | first M7-G1 |
| Truncated + corrected labels | 0.682 | 0.680 | 0.003 | M7-G1c |
| **Full-seq + graph bias** | **0.723** | **0.691** | **0.033** | final architecture |

Fixing artefacts raised both AUCs and recovered margin, but **still < 0.05** on
the architecture we ship. Pre-registered remedy applies: enable `return_aux`.

**D2 / D3.** D2 PASS (macro-F1 0.603 vs shuffled 0.181, margin 0.422; all six
templates active). D3 PASS. → `loop_aux` stays off.

**ReturnHead balance fix.** Fixation-level return-within-H=20 is **~81%**
positive (val diagnostic 81.4%; corpus 81.3%). Candidate shorter horizons
(`reports/return_horizon_balance.json`):

| H | pos rate | balance pos_weight (=neg/pos) |
|---:|---:|---:|
| 5 | 0.727 | 0.375 |
| 10 | 0.775 | 0.290 |
| 15 | 0.798 | 0.253 |
| **20** | **0.813** | **0.230** |

Even H=5 stays ~73% positive — shortening cannot reach ~50% without changing
the construct. **Chosen fix:** keep H=20 (aligned with D1) and set
`losses.return_aux.pos_weight: 0.23` on the BCE (`n_neg/n_pos`). Do **not**
retarget to visit boundaries (ablation #6).

**Config for retrain (owner-run).**
- Identical to `baseline_fullseq_1536_graphbias` otherwise
- `return_aux.enabled: true`, weight 0.5, `pos_weight: 0.23`
- `loop_aux` / contrastive off
- Tag: `baseline_fullseq_1536_graphbias_return_aux`
- Run root: `runs/m6_fullseq_graphbias_return_aux/` (leave no-aux reference intact)

**Paper caveats.**
1. Post-aux D1 re-run is expected to clear +0.05 because `return_aux` trains
   nearly the same mapping the D1 probe tests — it confirms the remedy took,
   not that the representation improved generally. Meaningful read-outs:
   predictive metrics hold/improve, and M8 prototypes sharpen.
2. SEMANTIC AP **0.150** (this run, n=45 854) is **not** comparable to **0.165**
   (truncated run, n=26 240): late-episode behaviour restored. **4× base-rate on
   the harder complete task is the GO.**

---

### 11) D1 closed — remedied to ceiling (M7-G3) (2026-07-22)

**Post-aux gate.** `runs/m6_fullseq_graphbias_return_aux/fold0_seed13`: D1
margin **0.0414** (emb 0.732 / feat 0.690) — still < 0.05, up from 0.0326
pre-aux. Full report: `reports/d1_ceiling_diagnostics.md`.

**Four diagnostics (no retrain).**

| Check | Result |
|---|---|
| ReturnHead val AUC | **0.734** |
| D1 emb probe AUC | **0.732** (corr with head scores **0.99**) |
| GBM ceiling (feat + history aggregates) | **0.745** (feat-only GBM 0.735; hist-only 0.680) |
| Balance fix shipped | **`pos_weight: 0.23`** at H=20; val pos rate **81.4%**; confirmed in frozen `train_config.yaml` |

**Alignment.** ReturnHead and D1 probe consume the **same** representation:
per-token `y = encode(batch)` (final transformer layer, **no pooling**). Head =
`Linear(d_model→1)`; probe = `StandardScaler + LogisticRegression(balanced)` on
the same `y`.

**Decision rule.** head ≈ probe ≈ ceiling → **close D1 as “remedied to
ceiling”**. Clearing +0.05 would need emb AUC ≥ 0.741, within ~0.004 of the
independent GBM ceiling and above the trained head — not a probe failure and
not a missing-balance failure. No escalation retrain.

**D1 margin trajectory (final):**

| Step | margin |
|---|---:|
| Truncated, buggy detector | 0.021 |
| Truncated, corrected labels | 0.003 |
| Full-seq + graph bias (no aux) | 0.033 |
| Full-seq + graph bias + return_aux (pos_weight 0.23) | **0.041** |

**Record.** Remedy applied as pre-registered on the ship architecture; margin
improved monotonically after validity fixes; **+0.05 exceeds the attainable
ceiling** given return-related fields are already model *inputs* (strong
feature-only / GBM baselines). Lesson: pre-registering absolute margins
against unknown task ceilings is brittle when the baseline channel already
carries the construct.

**What D1 actually discovered (paper sentence).** GBM on **features alone**
reaches **0.7355** — essentially matching the embedding probe (**0.732**).
So the fair scientific summary is *not* “embeddings add return information.”
It is: return predictability at H=20 is **almost entirely carried by the input
features nonlinearly combined**, tops out around **0.745** (feat+history GBM),
and what the embeddings contribute is **linearising** that information — a
linear probe on embeddings matches a boosted-tree model on raw features.
That is a real finding about examiner returns (substantially
recency/visit-feature-driven at this horizon), it explains why the +0.05 gate
was unpassable, and it is a stronger paper result than a bare D1 pass would
have been.

**`return_aux` stays on** (weight 0.5, pos_weight 0.23). Loop_aux remains off.
Ready for predictive confirmation and matrix launch under tag
`baseline_fullseq_1536_graphbias_return_aux` once the owner signs off.

---

## M4-A1 — Pure-torch edge-aware GAT (no PyG import) (2026-07-20)

**Decision.** M4 `CompactGNN` is implemented as a pure-PyTorch GATv2-style layer
(relation embeddings + edge attributes in attention), not `torch_geometric.nn.GATv2Conv`.

**Rationale.** `import torch_geometric` currently fails on this Windows host
(WinError 6714 via PyG→pandas→pyarrow filesystem transaction). Semantics match
research plan §7; PyG can be swapped later if the env is cleaned up.
