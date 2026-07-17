#!/usr/bin/env python3
"""Visual Gate overlay checker (Gate 1 alignment / Gate 2 assignment).

Self-contained Plotly HTML under reports/gaze_checks/gate{1,2}/.
Gate 2 requires owner sign-off in reports/DECISIONS.md before Stage 2 / M2+.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.viz.gate2_overlay import run_gate2_batch  # noqa: E402
from src.viz.overlay_check import run_gate1_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaze overlay checker (Gate 1 / Gate 2)")
    parser.add_argument("--gate", type=int, choices=[1, 2], default=2, help="Which visual gate")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build a fixture-based smoke HTML (no real _data required)",
    )
    args = parser.parse_args()
    if args.gate == 1:
        summary = run_gate1_batch(ROOT, smoke=bool(args.smoke))
        stop = "GATE 1 STOP: review reports/gaze_checks/gate1/index.html before P5."
    else:
        summary = run_gate2_batch(ROOT, smoke=bool(args.smoke))
        stop = (
            "GATE 2 HARD STOP: review reports/gaze_checks/gate2/index.html "
            "and sign off in reports/DECISIONS.md before any model code (M2+)."
        )
    print(json.dumps(summary, indent=2))
    if not args.smoke:
        print(f"\n*** {stop} ***\n")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
