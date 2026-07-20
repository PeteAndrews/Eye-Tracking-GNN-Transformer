# Encoder eval pairs (M2 human gate)

Active categories only: `response_mark_scheme`, `commentary_paraphrase`
(command_word / level_descriptor retired — see DECISIONS.md M2-A2).

Draft mixes **hard** (`hard_within_trial`) and **easy** (`easy_cross_trial`) ~50/50.

1. Open `draft_pairs_v1.csv` as **UTF-8**.
2. Rows with `reviewed=true` are already accepted keepers — leave them.
3. Review only `reviewed=false` rows: keep → set `reviewed=true`; reject → delete or leave false.
4. Related should beat unrelated for the anchor. Hard unrelated = same-trial content distractor.
5. Target ≥20 `reviewed=true`, then:

```text
python scripts/promote_encoder_pairs.py
python scripts/run_encoder_bakeoff.py --freeze
```

Bake-off reports overall / hard / easy ranking accuracy; hard is the tie-breaker.
