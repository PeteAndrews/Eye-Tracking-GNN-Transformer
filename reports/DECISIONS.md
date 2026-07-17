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

## P4 — Visual Gate 1 sign-off (PENDING)

**Status:** tooling + stratified sample generated (`reports/gaze_checks/gate1/`).
**Owner action required:** review `index.html` and the episode HTMLs; triage any
misalignment (metadata / identity / coordinates); then replace this section with
an explicit sign-off (or a punch-list of fixes) before P5 may start.

```
# Owner fills in after review — example:
# ## P4 — Visual Gate 1 sign-off (YYYY-MM-DD)
# Reviewed stratified Gate 1 sample (75 episodes). Alignment acceptable.
# Signed off for P5. — <name>
```
