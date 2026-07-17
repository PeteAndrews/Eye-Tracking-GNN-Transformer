# Data QC

Accumulating preprocessing QC notes (P0–P7).

---

## P0 — Registries and identity (2026-07-17)

- Metadata variants: **36**; document images: **36**; trials: **30**.
- Star assignments: **750** (25 participants × 30 trials).
- Star-on count: **exactly 3 per participant** among T11/T12/T13/T21/T27/T30 — validated.
- Question types: **30/30** trials constant across participants (source: gaze `Question type`).
- `question_id` stored as `trial_id` (one response per question).
- Variant consistency: see `reports/DECISIONS.md` **P0-V1** (segment asymmetries triaged; geometry soft-drift expected).
- Outputs: `data_processed/v0_p0/registry/`.
