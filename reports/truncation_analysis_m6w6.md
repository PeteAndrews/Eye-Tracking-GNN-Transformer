# Truncation analysis (M6-W6) — cap 256 vs full corpus

Source: `data_processed/v0_p0/fixations/episode_qc.parquet` (750 episodes)
plus per-episode fixation parquets for loop-template row positions.
JSON: `reports/truncation_analysis_m6w6.json`.

## Overall (cap = 256)

| Metric | Value |
|---|---|
| Episodes with `n_fixations > 256` | **43.1%** (323 / 750) |
| Fixations discarded | **40.2%** (88 014 / 219 042) |
| Mean / median length | 292.1 / **206.5** |
| p75 / p90 / p95 / max | 428.5 / 684.4 / 822.6 / **1520** |

Note: median (206.5) is below the cap, so a **minority** of episodes are
truncated — but those are the long ones, so **two-fifths of all fixation
mass** never reaches the model. Mean > median because of a heavy right tail.

## By `star_condition`

| Condition | n | median | p90 | max | % eps >256 | % fix lost |
|---|---:|---:|---:|---:|---:|---:|
| `star_on` | 75 | **526** | 924 | 1520 | **89.3%** | **56.8%** |
| `star_off` | 75 | **526** | 912 | 1261 | **89.3%** | **55.8%** |
| `not_eligible` | 600 | 155 | 490 | 1506 | 31.5% | 29.8% |

Star-eligible trials (on or off) are ~2× longer than other trials. The 256 cap
disproportionately guts the star natural-experiment stratum.

## Episode-length coverage at candidate caps

| Cap | % episodes fully covered |
|---:|---:|
| 256 | 56.9% |
| 512 | 82.0% |
| 768 | 93.7% |
| 1024 | 98.3% |
| 1280 | 99.3% |
| **1536** | **100%** (covers max 1520) |

## Star / LD decomposition

| | Star template rows | Level-descriptor rows |
|---|---:|---:|
| n events (annotated rows) | 1283 | 2929 |
| % with absolute index ≥ 256 | **64.2%** | 59.1% |
| % with relative `t/T ≥ 0.5` | **74.8%** | 58.7% |

| Length | `star_on` | other (`star_off` ∪ `not_eligible`) |
|---|---:|---:|
| mean | **579.5** | 260.1 |
| median | **526** | 175 |
| max | 1520 | 1506 |

**Interpretation.** Both mechanisms operate:

1. **`star_on` episodes are simply long** (median 526 vs 175) → absolute indices
   past 256 even for mid-episode behaviour.
2. **Star loops are also late in relative time** (75% in the second half of the
   episode vs 59% for LD) → marking-phase star use is concentrated after
   orientation/reading.

M8 star analysis under cap 256 therefore sees roughly one-third of star-loop
rows, biased toward early absolute time.
