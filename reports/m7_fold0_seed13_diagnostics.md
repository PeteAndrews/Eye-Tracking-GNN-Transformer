# M7 diagnostic gate (frozen M6 embeddings)

- Checkpoint: `D:\Projects\GNN-Transformer-Eye-Tracking\runs\m6\fold0_seed13\checkpoint_best.pt`
- Fold **0** · seed **13**
- Val participants: P03, P06, P10, P12, P21
- Episodes: train 600 / val 150

## Decision: **FAIL**

FAIL — enable return/loop aux losses and retrain (see DECISIONS)

| gate | pass | metric |
|---|---|---|
| D1 return | False | emb AUC=0.7002 · feat=0.6795 · margin=0.0207 (need ≥ 0.05) |
| D2 loop template | True | macro-F1=0.7070 · shuffled=0.2753 · margin=0.4317 (need ≥ 0.05) |
| D3 subsequence | True | AUC=0.6314985209659485 · acc=0.5952937509739754 |

## D2 templates

Active: `response→mark_scheme→response`, `response→commentary→response`, `mark_scheme→response→mark_scheme`, `question→response→question`

Dropped (< min count):
- `response→mark_scheme_level_descriptor→response` count=0
- `response→star_chart→response` count=0

## Fixation vs visit (diagnostic slice)

- Fixation steps: n=23460 · return-within-H rate=0.8236572890025575
- Visit boundaries: n=9244 · return-within-H rate=0.5524664647338814
- Visit-boundary = last fixation of each contiguous same-segment run. Full visit-token retrain is ablation #6; this table is the M7 diagnostic slice.

## D1 ROC

![D1 ROC](d1_roc.svg)

