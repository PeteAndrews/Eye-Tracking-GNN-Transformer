#!/usr/bin/env python
"""Freeze / re-freeze TextEncoderV1 from a candidate id (bake-off override)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text.encoder import freeze_encoder_id
from src.utils import io as uio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--encoder-id", required=True, help="Candidate id, e.g. bge_large")
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Selection rationale recorded on the card.",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    bakeoff_path = root / "reports" / "encoder_bakeoff_v1.json"
    summary = uio.read_json(bakeoff_path) if bakeoff_path.is_file() else None
    path = freeze_encoder_id(
        root,
        args.encoder_id,
        bakeoff_summary=summary,
        selection_note=args.note or None,
    )
    card = uio.read_json(path)
    print(json.dumps({"frozen_card": str(path), "card": card}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
