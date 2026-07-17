#!/usr/bin/env python3
"""Run P2 metadata compilation (audit gate + segment/panel tables)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.segments import run_p2  # noqa: E402


def main() -> int:
    summary = run_p2(ROOT, require_audit=True)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
