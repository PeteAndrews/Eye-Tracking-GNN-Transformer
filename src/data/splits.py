"""Grouped participant K-fold splits (sole train/tune/ablation protocol)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


def list_participants(episode_keys: Sequence[tuple[str, ...]]) -> list[str]:
    """Unique participant ids preserving first-seen order."""
    seen: list[str] = []
    for key in episode_keys:
        pid = str(key[0])
        if pid not in seen:
            seen.append(pid)
    return seen


def grouped_participant_folds(
    episode_keys: Sequence[tuple[str, ...]],
    *,
    n_folds: int = 5,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Return fold dicts with train/val index lists over episodes.

    ``episode_keys[i][0]`` must be ``participant_id``. Groups never split
    across train/val. ``seed`` only shuffles participant order before
    assigning fold ids (GroupKFold itself is deterministic given groups).
    """
    if len(episode_keys) == 0:
        return []
    groups = np.array([str(k[0]) for k in episode_keys])
    n_groups = len(set(groups.tolist()))
    n_splits = min(int(n_folds), n_groups)
    if n_splits < 2:
        raise ValueError(f"Need ≥2 participants for grouped folds; got {n_groups}")

    # Stable shuffle of fold assignment via participant order permutation
    rng = np.random.default_rng(seed)
    participants = list_participants(episode_keys)
    order = rng.permutation(len(participants))
    pid_rank = {participants[i]: int(r) for r, i in enumerate(order)}
    # Remap groups to rank for a seeded GroupKFold partition
    group_ids = np.array([pid_rank[g] for g in groups])

    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(len(episode_keys))
    folds: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(gkf.split(X, groups=group_ids)):
        tr_p = sorted({groups[i] for i in tr})
        va_p = sorted({groups[i] for i in va})
        assert set(tr_p).isdisjoint(va_p)
        folds.append(
            {
                "fold": fold_id,
                "train_idx": tr.tolist(),
                "val_idx": va.tolist(),
                "train_participants": tr_p,
                "val_participants": va_p,
            }
        )
    return folds


def leave_one_question_out_folds(
    episode_keys: Sequence[tuple[str, str, ...]],
    *,
    question_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Post-hoc LOQO only — not for tuning. ``episode_keys[i][1]`` unused;
    pass parallel ``question_ids`` aligned to episodes.
    """
    if len(episode_keys) != len(question_ids):
        raise ValueError("question_ids must align with episode_keys")
    q_unique = sorted(set(str(q) for q in question_ids))
    folds = []
    for fold_id, q_hold in enumerate(q_unique):
        tr = [i for i, q in enumerate(question_ids) if str(q) != q_hold]
        va = [i for i, q in enumerate(question_ids) if str(q) == q_hold]
        folds.append(
            {
                "fold": fold_id,
                "held_out_question": q_hold,
                "train_idx": tr,
                "val_idx": va,
                "post_hoc_only": True,
            }
        )
    return folds
