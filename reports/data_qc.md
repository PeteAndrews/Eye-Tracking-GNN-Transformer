# Data QC

Accumulating preprocessing QC notes (P0–P7).

---

## P0 — Registries and identity (2026-07-17)

- Metadata variants: **36**; document images: **36**; trials: **30**.
- Star assignments: **750** (25 participants × 30 trials).
- Star-on count: **exactly 3 per participant** among T11/T12/T13/T21/T27/T30 — validated.
- Question types: **30/30** trials constant across participants (source: gaze `Question type`).
- `question_id` stored as `trial_id` (one response per question).
- Dimension registry **rebuilt** after owner replaced NS images (still 1920×1080 for NS — consistent with shorter non-star UI; boxes redrawn to fit).
- Variant correspondence (M3-C1): T12/T13/T21/T27/**T30 PASS**; **T11 FAIL** (empty NS text + commentary split). See DECISIONS.md.
- Outputs: `data_processed/v0_p0/registry/`.

## P1 — Gaze prune/tidy

- Participants written: 25/25
- Episodes: 750
- Rows with correction_applied=False (trusted, counted): 8601268
- Outputs: `data_processed/v0_p0/gaze_pruned/`

## P2.6 audit (re-run after owner fix)

- **36/36 PASS** after T11NS empty-text fix and path flatten (2026-07-17 afternoon).
- All previous NS `bbox_outside_document` errors cleared.
- Report: `reports/metadata_audit/`

## P0 correspondence (M3-C1)

- All six eligible trials PASS after allowlisting remaining T11 star-instruction commentary fragments.
- Soft geometry/AOI-id drift only (expected under per-variant construction).

## P1 — Gaze prune/tidy

- Participants written: 25/25
- Episodes: 750
- Rows with correction_applied=False (trusted, counted): 8601268
- Outputs: `D:\Projects\GNN-Transformer-Eye-Tracking\data_processed\v0_p0\gaze_pruned`

## P0 registry validation

- ERROR: Missing metadata for stem T01
- ERROR: Missing metadata for stem T02
- ERROR: Missing metadata for stem T03
- ERROR: Missing metadata for stem T04
- ERROR: Missing metadata for stem T05
- ERROR: Missing metadata for stem T06
- ERROR: Missing metadata for stem T07
- ERROR: Missing metadata for stem T08
- ERROR: Missing metadata for stem T09
- ERROR: Missing metadata for stem T10
- ERROR: Missing metadata for stem T11NS
- ERROR: Missing metadata for stem T11S
- ERROR: Missing metadata for stem T12NS
- ERROR: Missing metadata for stem T12S
- ERROR: Missing metadata for stem T13NS
- ERROR: Missing metadata for stem T13S
- ERROR: Missing metadata for stem T14
- ERROR: Missing metadata for stem T15
- ERROR: Missing metadata for stem T16
- ERROR: Missing metadata for stem T17
- ERROR: Missing metadata for stem T18
- ERROR: Missing metadata for stem T19
- ERROR: Missing metadata for stem T20
- ERROR: Missing metadata for stem T21NS
- ERROR: Missing metadata for stem T21S
- ERROR: Missing metadata for stem T22
- ERROR: Missing metadata for stem T23
- ERROR: Missing metadata for stem T24
- ERROR: Missing metadata for stem T25
- ERROR: Missing metadata for stem T26
- ERROR: Missing metadata for stem T27NS
- ERROR: Missing metadata for stem T27S
- ERROR: Missing metadata for stem T28
- ERROR: Missing metadata for stem T29
- ERROR: Missing metadata for stem T30NS
- ERROR: Missing metadata for stem T30S
- ERROR: Variant consistency T11: Missing S/NS pair: S=None, NS=None
- ERROR: Variant consistency T12: Missing S/NS pair: S=None, NS=None
- ERROR: Variant consistency T13: Missing S/NS pair: S=None, NS=None
- ERROR: Variant consistency T21: Missing S/NS pair: S=None, NS=None
- ERROR: Variant consistency T27: Missing S/NS pair: S=None, NS=None
- ERROR: Variant consistency T30: Missing S/NS pair: S=None, NS=None

## P0 registry validation

- ERROR: Variant consistency T11: AOI identity differ in ['aoi_id_types']; segment triage ['segments']; soft drift in ['aoi_geometry', 'text_boxes']
- ERROR: Variant consistency T12: AOI identity differ in ['aoi_id_types']; segment triage ['segments']; soft drift in ['aoi_geometry', 'text_boxes']

## P0 registry validation

- ERROR: Variant consistency T11: 2 unmatched NS segments; 4 unexpected S-only segments; 1 allowlisted star-conditional S-only; within-panel order mismatch: ['commentary']; soft drift in ['aoi_geometry', 'text_boxes', 'aoi_id_types']
- ERROR: Variant consistency T21: 1 unexpected S-only segments; soft drift in ['aoi_geometry', 'text_boxes']

## P0 registry validation

- ERROR: Variant consistency T11: 2 unmatched NS segments; 3 unexpected S-only segments; 2 allowlisted star-conditional S-only; within-panel order mismatch: ['commentary']; soft drift in ['aoi_geometry', 'text_boxes', 'aoi_id_types']

## P0 registry validation

- ERROR: Variant consistency T11: 1 unexpected S-only segments; 2 allowlisted star-conditional S-only; soft drift in ['aoi_geometry', 'text_boxes']

