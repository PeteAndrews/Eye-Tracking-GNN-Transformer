#!/usr/bin/env python3
"""Run P5 coordinate finalisation (DOCnorm + viewport features)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coords import run_p5  # noqa: E402


def main() -> int:
    summary = run_p5(ROOT)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
