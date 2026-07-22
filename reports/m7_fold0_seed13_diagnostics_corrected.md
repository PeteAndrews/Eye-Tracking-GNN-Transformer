# M7 diagnostic gate (frozen M6 embeddings)

- Checkpoint: `D:\Projects\GNN-Transformer-Eye-Tracking\runs\m6\fold0_seed13\checkpoint_best.pt`
- Fold **0** · seed **13**
- Val participants: P03, P06, P10, P12, P21
- Episodes: train 600 / val 150

## Decision: **FAIL**

FAIL — enable return/loop aux losses and retrain (see DECISIONS)

| gate | pass | metric |
|---|---|---|
| D1 return | False | emb AUC=0.6822 · feat=0.6795 · margin=0.0027 (need ≥ 0.05) |
| D2 loop template | True | macro-F1=0.5326 · shuffled=0.2143 · margin=0.3183 (need ≥ 0.05) |
| D3 subsequence | True | AUC=0.8719134922142316 · acc=0.81369798971482 |

## D2 templates

Active: `response→mark_scheme→response`, `response→mark_scheme_level_descriptor→response`, `response→commentary→response`, `mark_scheme→response→mark_scheme`, `question→response→question`, `response→star_chart→response`

## Fixation vs visit (diagnostic slice)

- Fixation steps: n=23460 · return-within-H rate=0.8236572890025575
- Visit boundaries: n=9244 · return-within-H rate=0.5524664647338814
- Visit-boundary = last fixation of each contiguous same-segment run. Full visit-token retrain is ablation #6; this table is the M7 diagnostic slice.

## D1 ROC

![D1 ROC](d1_roc.svg)

## D2 corpus template counts (corrected re-detect)

- `response→mark_scheme→response`: **16909**
- `mark_scheme→response→mark_scheme`: **11665**
- `question→response→question`: **3250**
- `response→commentary→response`: **1620**
- `response→mark_scheme_level_descriptor→response`: **907**
- `response→star_chart→response`: **346**

Note: counts are over fold0 train+val episodes truncated to max_seq_len; corpus-wide full-length star triples are ~1001.

Supersedes `reports/m7_fold0_seed13_diagnostics.md` (buggy re-detect). Training annotations unchanged (P6 parquet).
