#!/usr/bin/env python3
"""Run P6 fixation construction (aggregation, assignment, loops)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fixations import run_p6  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-participants", type=int, default=None)
    args = p.parse_args()
    summary = run_p6(ROOT, max_participants=args.max_participants)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
