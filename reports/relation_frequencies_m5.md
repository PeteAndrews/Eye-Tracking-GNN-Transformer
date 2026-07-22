# M5 next-relation label frequencies

- Episodes counted: **750** (skipped 0)
- Transitions: **218292**
- Graph version: `g1_bge1024`
- Throughput: **45.37** episodes/s

| relation | count | freq | inv_freq_weight |
|---|---:|---:|---:|
| `NEXT_SEGMENT` | 22003 | 0.1008 | 9.921 |
| `PREVIOUS_SEGMENT` | 16408 | 0.0752 | 13.304 |
| `BELONGS_TO` | 0 | 0.0000 | — |
| `SPATIAL_NEIGHBOUR` | 50571 | 0.2317 | 4.3165 |
| `SEMANTIC_CANDIDATE` | 8418 | 0.0386 | 25.9316 |
| `NO_DIRECT_RELATION` | 134831 | 0.6177 | 1.619 |
| `EMPTY_SPACE_TRANSITION` | 22876 | 0.1048 | 9.5424 |

Weights above are raw inverse frequency (`n_trans / count`). **M6 locked (DECISIONS M6-PRE / M6-W1):** use this table as-is (no per-fold recompute); clip at `clip_max: 10.0` in `configs/train.yaml`; exclude zero-count labels (`BELONGS_TO`) from the BCE head.

