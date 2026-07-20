#!/usr/bin/env python
"""Run M2 encoder bake-off on reviewed pairs; optionally freeze TextEncoderV1."""

from __future__ import annotations

import os

# Must be set before huggingface_hub / transformers weight fetches (Windows SIGILL via hf_xet).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("bakeoff: starting", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Score draft pairs without review gate (diagnostics only; not for freezing).",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write TextEncoderV1 card from winner (requires reviewed bake-off).",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    require = not args.allow_unreviewed

    from src.text.encoder_selection import run_bakeoff

    print("bakeoff: running", flush=True)
    summary = run_bakeoff(root, require_reviewed=require)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if not summary.get("ok"):
        return 1
    if args.freeze:
        if args.allow_unreviewed:
            print("Refusing --freeze with --allow-unreviewed", file=sys.stderr)
            return 1
        from src.text.encoder import freeze_winner

        path = freeze_winner(root, summary)
        print(json.dumps({"frozen_card": str(path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
