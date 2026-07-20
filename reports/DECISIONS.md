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

## M4-A1 — Pure-torch edge-aware GAT (no PyG import) (2026-07-20)

**Decision.** M4 `CompactGNN` is implemented as a pure-PyTorch GATv2-style layer
(relation embeddings + edge attributes in attention), not `torch_geometric.nn.GATv2Conv`.

**Rationale.** `import torch_geometric` currently fails on this Windows host
(WinError 6714 via PyG→pandas→pyarrow filesystem transaction). Semantics match
research plan §7; PyG can be swapped later if the env is cleaned up.
