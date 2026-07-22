# M6 predictive eval (grouped-val)

- Checkpoint: `D:\Projects\GNN-Transformer-Eye-Tracking\runs\m6\fold0_seed13\checkpoint_best.pt`
- Fold: **0** · seed used for dataset RNG: 13
- Val participants: P03, P06, P10, P12, P21
- Relation steps scored: **26240**
- Operating threshold (sigmoid): **0.5**
- **Go/no-go: GO** (SEMANTIC_CANDIDATE AP=0.164535 vs baseline=0.037043)

## Per-label next-relation

| relation | n_pos | base_rate | AP | AP baseline | Δ | P@thr | R@thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| `NEXT_SEGMENT` | 2392 | 0.0912 | 0.19158 | 0.0912 | 0.100421 | 0.156304 | 0.713629 |
| `PREVIOUS_SEGMENT` | 1643 | 0.0626 | 0.18172 | 0.0626 | 0.119106 | 0.144596 | 0.665247 |
| `SPATIAL_NEIGHBOUR` | 5393 | 0.2055 | 0.347345 | 0.2055 | 0.141819 | 0.296241 | 0.812535 |
| `SEMANTIC_CANDIDATE` | 972 | 0.0370 | 0.164535 | 0.0370 | 0.127493 | 0.135998 | 0.556584 |
| `NO_DIRECT_RELATION` | 15798 | 0.6021 | 0.784665 | 0.6021 | 0.182608 | 0.680973 | 0.983226 |
| `EMPTY_SPACE_TRANSITION` | 3909 | 0.1490 | 0.861311 | 0.1490 | 0.71234 | 0.552627 | 0.831415 |

## Ranking

- **model**: MRR=0.7891 · hits@1=0.6762 · hits@3=0.8790 · hits@5=0.9402 · n=23433
- **transition_frequency_baseline**: MRR=0.4569 · hits@1=0.2773 · hits@3=0.5120 · hits@5=0.6810 · n=23433
- **feature_only_cosine_probe**: MRR=0.7886 · hits@1=0.7251 · hits@3=0.8026 · hits@5=0.8645 · n=23433

## Next-panel

- n=26240 · accuracy=0.8271 · macro-F1=0.7777

Per-class F1:

- `question`: 0.8295
- `response`: 0.8075
- `mark_scheme`: 0.7941
- `commentary`: 0.9197
- `star_chart`: 0.7059
- `ui`: 0.0000
- `outside_document`: 0.6092

Confusion matrix (rows=true, cols=pred):

```
      ques  resp  mark  comm  star    ui  outs
ques  3168   239   197    46    11     0   163
resp   192  4759   573    62    27     0   138
mark   244   710  5288   351    13     0   144
comm    44    78   326  7254    45     0   148
star     3    15    16    16   198     0     8
  ui     0     0     0     0     0     0     0
outs   163   235   169   150    11     0  1036
```

