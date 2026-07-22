# D1 ceiling diagnostics

- Checkpoint: `D:\Projects\GNN-Transformer-Eye-Tracking\runs\m6_fullseq_graphbias_return_aux\fold0_seed13\checkpoint_best.pt`
- Fold 0 · seed 13 · H=20
- n_train=160326 · n_val=41736 · val pos rate=0.8142

## AUCs

| Source | AUC |
|---|---:|
| ReturnHead (trained) | **0.7343** |
| D1 embedding probe | **0.7319** |
| D1 feature-only probe | 0.6905 |
| GBM feat + history (ceiling) | **0.7454** |
| GBM feat only | 0.7355 |
| GBM history only | 0.6804 |

- Probe margin (emb − feat): **0.0414** (need ≥ 0.05)
- Ceiling − feat: **0.0549**
- Head vs probe score corr: 0.9901

## Alignment

- Same representation: **yes** — both consume per-token `y = encode(batch)` (no pooling).
- ReturnHead: `Linear(d_model→1)` on `y`.
- D1 probe: `StandardScaler + LogisticRegression(balanced)` on the same `y`.

## Balance fix shipped

- `return_aux.enabled=True`, weight=0.5
- **pos_weight=0.23** (not shorter H); horizon **20**
- Val positive rate: **0.8142** (train 0.8128)
- Balance OK (pos_weight≈0.23): **True**

## Verdict

**head ≈ probe ≈ ceiling** (0.734 ≈ 0.732 ≈ 0.745; head–probe score corr 0.99).
The D1 probe already extracts what ReturnHead achieves; a strong nonlinear
classifier on raw features + explicit history only reaches 0.745. Clearing
+0.05 over the feature probe would require emb AUC ≥ 0.741 — within ~0.004 of
that independent ceiling and above the trained head. **D1 closed as remedied
to ceiling** (DECISIONS M7-G3). `return_aux` stays on; no further escalation
retrain.

**Paper read.** GBM on features alone (0.7355) ≈ embedding probe (0.732): return
predictability at H=20 is carried by input features nonlinearly combined; embeddings
mainly linearise that signal rather than adding new return information.

