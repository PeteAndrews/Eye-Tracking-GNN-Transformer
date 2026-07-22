"""Corpus truncation accounting for run summaries (standing validity guard)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from src.utils.arrow_cuda import read_parquet


def truncation_stats_from_qc(
    episode_qc_path: Path,
    *,
    max_seq_len: int,
    keys: Optional[Sequence[tuple[str, str, str]]] = None,
) -> dict[str, Any]:
    """Return truncated episode/fixation counts at ``max_seq_len``.

    If ``keys`` is given (participant, trial, star_condition), restrict to that
    subset (e.g. fold train∪val); otherwise use the full QC table.
    """
    qc = read_parquet(Path(episode_qc_path))
    if keys is not None:
        keyset = {(str(p), str(t), str(s)) for p, t, s in keys}
        mask = [
            (str(r.participant_id), str(r.trial_id), str(r.star_condition)) in keyset
            for r in qc.itertuples(index=False)
        ]
        qc = qc.loc[mask]
    lengths = qc["n_fixations"].to_numpy(dtype=np.int64)
    n = int(lengths.size)
    if n == 0:
        return {
            "max_seq_len": int(max_seq_len),
            "n_episodes": 0,
            "n_episodes_truncated": 0,
            "frac_episodes_truncated": 0.0,
            "n_fixations_total": 0,
            "n_fixations_kept": 0,
            "n_fixations_discarded": 0,
            "frac_fixations_discarded": 0.0,
        }
    cap = int(max_seq_len)
    truncated = lengths > cap
    total = int(lengths.sum())
    kept = int(np.minimum(lengths, cap).sum())
    discarded = total - kept
    return {
        "max_seq_len": cap,
        "n_episodes": n,
        "n_episodes_truncated": int(truncated.sum()),
        "frac_episodes_truncated": float(truncated.mean()),
        "n_fixations_total": total,
        "n_fixations_kept": kept,
        "n_fixations_discarded": discarded,
        "frac_fixations_discarded": float(discarded / total) if total else 0.0,
        "length_median": float(np.median(lengths)),
        "length_max": int(lengths.max()),
    }


def truncation_stats_for_keys(
    repo: Path,
    keys: Iterable[tuple[str, str, str]],
    *,
    max_seq_len: int,
    fixations_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Prefer episode_qc.parquet; fall back to counting parquet rows per key."""
    repo = Path(repo)
    qc_path = repo / "data_processed" / "v0_p0" / "fixations" / "episode_qc.parquet"
    key_list = list(keys)
    if qc_path.is_file():
        return truncation_stats_from_qc(qc_path, max_seq_len=max_seq_len, keys=key_list)

    root = (
        Path(fixations_root)
        if fixations_root
        else repo / "data_processed" / "v0_p0" / "fixations"
    )
    lengths: list[int] = []
    for pid, tid, sc in key_list:
        pq = root / pid / f"{tid}__{sc}.parquet"
        if pq.is_file():
            lengths.append(len(read_parquet(pq)))
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    cap = int(max_seq_len)
    empty = {
        "max_seq_len": cap,
        "n_episodes": 0,
        "n_episodes_truncated": 0,
        "frac_episodes_truncated": 0.0,
        "n_fixations_total": 0,
        "n_fixations_kept": 0,
        "n_fixations_discarded": 0,
        "frac_fixations_discarded": 0.0,
    }
    if lengths_arr.size == 0:
        return empty
    truncated = lengths_arr > cap
    total = int(lengths_arr.sum())
    kept = int(np.minimum(lengths_arr, cap).sum())
    return {
        "max_seq_len": cap,
        "n_episodes": int(lengths_arr.size),
        "n_episodes_truncated": int(truncated.sum()),
        "frac_episodes_truncated": float(truncated.mean()),
        "n_fixations_total": total,
        "n_fixations_kept": kept,
        "n_fixations_discarded": total - kept,
        "frac_fixations_discarded": float((total - kept) / total) if total else 0.0,
        "length_median": float(np.median(lengths_arr)),
        "length_max": int(lengths_arr.max()),
    }
