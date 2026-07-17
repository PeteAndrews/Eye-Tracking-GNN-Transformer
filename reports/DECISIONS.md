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

**Path note.** Audited JSONs currently live at
`_data/annotations-audited/annotations-audited/complete/` (nested). Config
updated. `audit_metadata.py` is at `_data/annotations-audited-legacy/`.
`T10-completee.json` typo reappeared in the new export — rename again when convenient.
