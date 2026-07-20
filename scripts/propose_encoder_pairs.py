#!/usr/bin/env python
"""Propose draft encoder eval pairs (M2). Supports full regen or append-to-keepers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text.encoder_selection import append_new_to_keepers, propose_and_write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--append-to",
        type=Path,
        default=None,
        help="Owner-reviewed keepers CSV (all reviewed=true). Appends ~n-new candidates.",
    )
    parser.add_argument(
        "--n-new",
        type=int,
        default=15,
        help="Number of new unreviewed candidates to append (default 15).",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.append_to is not None:
        path = args.append_to
        if not path.is_absolute():
            path = root / path
        result = append_new_to_keepers(root, path, n_new=args.n_new)
    else:
        result = propose_and_write(root)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
