#!/usr/bin/env python3
"""Run P1 gaze pruning on all participant TSVs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gaze_load import run_p1
from src.utils import io as uio


def main() -> int:
    summary = run_p1(ROOT)
    print(f"P1 out: {summary['out_dir']}")
    print(
        f"files={summary['n_files']} written={summary['n_written']} "
        f"episodes={summary['n_episodes']} "
        f"correction_false_rows={summary['n_correction_false_total']}"
    )
    if summary["errors"]:
        print("ERRORS:")
        for e in summary["errors"]:
            print(f"  - {e}")
        return 1

    # Append QC snapshot to reports/data_qc.md
    qc_path = ROOT / "reports" / "data_qc.md"
    existing = uio.read_text(qc_path) if qc_path.is_file() else "# Data QC\n\n"
    block = (
        "\n## P1 — Gaze prune/tidy\n\n"
        f"- Participants written: {summary['n_written']}/{summary['n_files']}\n"
        f"- Episodes: {summary['n_episodes']}\n"
        f"- Rows with correction_applied=False (trusted, counted): "
        f"{summary['n_correction_false_total']}\n"
        f"- Outputs: `{summary['out_dir']}`\n"
    )
    if "## P1 — Gaze prune/tidy" not in existing:
        uio.write_text(qc_path, existing.rstrip() + "\n" + block + "\n")
    print("P1 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
