# M7 diagnostic gate (frozen M6 embeddings)

- Checkpoint: `runs\m6_fullseq_graphbias\fold0_seed13\checkpoint_best.pt`
- Fold **0** · seed **13**
- Val participants: P03, P06, P10, P12, P21
- Episodes: train 600 / val 150

## Decision: **FAIL**

FAIL — enable return/loop aux losses and retrain (see DECISIONS)

| gate | pass | metric |
|---|---|---|
| D1 return | False | emb AUC=0.7231 · feat=0.6905 · margin=0.0326 (need ≥ 0.05) |
| D2 loop template | True | macro-F1=0.6025 · shuffled=0.1810 · margin=0.4215 (need ≥ 0.05) |
| D3 subsequence | True | AUC=0.5803564527634191 · acc=0.5544142389090587 |

## D2 templates

Active: `response→mark_scheme→response`, `response→mark_scheme_level_descriptor→response`, `response→commentary→response`, `mark_scheme→response→mark_scheme`, `question→response→question`, `response→star_chart→response`

## Fixation vs visit (diagnostic slice)

- Fixation steps: n=41736 · return-within-H rate=0.814165229058846
- Visit boundaries: n=17202 · return-within-H rate=0.5491221950935937
- Visit-boundary = last fixation of each contiguous same-segment run. Full visit-token retrain is ablation #6; this table is the M7 diagnostic slice.

## D1 ROC

![D1 ROC](d1_roc.svg)

