#!/usr/bin/env python3
"""Build P0 registries from real _data inputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.registry import run_p0
from src.utils import io as uio


def main() -> int:
    summary = run_p0(ROOT)
    print(f"P0 out: {summary['out_dir']}")
    print(
        f"variants={summary['n_trial_variants']} images={summary['n_document_images']} "
        f"star_assignments={summary['n_star_assignments']} "
        f"question_types={summary['n_question_types']} "
        f"variant_ok={summary['variant_ok']}"
    )
    if summary["errors"]:
        print(f"ERRORS ({len(summary['errors'])}):")
        for e in summary["errors"]:
            print(f"  - {e}")
        qc_path = ROOT / "reports" / "data_qc.md"
        existing = uio.read_text(qc_path) if qc_path.is_file() else "# Data QC\n\n"
        block = "\n## P0 registry validation\n\n"
        for e in summary["errors"]:
            block += f"- ERROR: {e}\n"
        uio.write_text(qc_path, existing.rstrip() + "\n" + block + "\n")
        return 1
    if summary.get("soft_warnings"):
        print(f"SOFT WARNINGS ({len(summary['soft_warnings'])}):")
        for w in summary["soft_warnings"]:
            print("  -", w.encode("ascii", "replace").decode("ascii"))
    print("P0 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
