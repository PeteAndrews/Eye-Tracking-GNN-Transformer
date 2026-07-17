# Decisions log

Gate sign-offs (owner), fallback activations, and documented deviations from the
pre-registered plan. Sign-offs are written by the owner, not the coding agent.

---

## P0-V1 — S/NS metadata variant consistency (2026-07-17)

**Finding.** After stripping star-chart AOIs, `star_concept` / `aoi_type=star_chart`
segments, and their text boxes, the six eligible trials are **not** byte-identical
in non-star content:

| Trial | AOI id/types | Segments | Geometry / boxes |
|---|---|---|---|
| T11 | match | match | soft drift |
| T12 | match | +1 commentary on S only (“enter a level…”) | soft drift |
| T13 | match | match | soft drift |
| T21 | match | +1 commentary on S only | soft drift |
| T27 | match | +1 commentary on S only | soft drift |
| T30 | match | asymmetric: NS has star-instruction commentary in `commentary`; S has same response bullet under a different `segment_id` | soft drift |

**Interpretation.** Soft AOI/box coordinate drift is expected when the star chart
changes layout. Segment asymmetries are annotation/placement differences
(star-instruction text lives under `commentary` on NS and partly under
`star_chart` on S; T30 renumbers a response bullet).

**P0 ruling (triaged).** AOI id/type mismatches remain hard failures. Segment
asymmetries and geometry drift are recorded as warnings and do **not** block P0.
Outputs land under `data_processed/v0_p0/registry/variant_consistency.json`.

**Open for owner before M3.** The frozen commitment that `star_on` is base +
overlay with an identical non-star subgraph needs a concrete base definition on
this corpus, e.g. (a) NS-canonical base + S star overlay, (b) intersection of
segment ids, or (c) metadata repair so S/NS non-star segments match. **Do not
implement M3 until the owner picks one.**
