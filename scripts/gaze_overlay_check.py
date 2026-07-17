#!/usr/bin/env python3
"""P4 Visual Gate 1 — metadata–gaze alignment overlays.

Self-contained Plotly HTML reports under reports/gaze_checks/gate1/.
Owner sign-off in reports/DECISIONS.md is required before P5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.viz.overlay_check import run_gate1_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 Visual Gate 1 overlay checker")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build a fixture-based smoke HTML (no real _data required)",
    )
    args = parser.parse_args()
    summary = run_gate1_batch(ROOT, smoke=bool(args.smoke))
    print(json.dumps(summary, indent=2))
    if not args.smoke:
        print(
            "\n*** GATE 1 STOP: review reports/gaze_checks/gate1/index.html "
            "and record sign-off in reports/DECISIONS.md before P5. ***\n"
        )
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
