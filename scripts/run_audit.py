#!/usr/bin/env python3
"""Invoke the existing P2.6 metadata audit tool (do not rewrite it)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    script = ROOT / str(cfg.paths.metadata_audit_script)
    meta = ROOT / str(cfg.paths.metadata_dir)
    images = ROOT / str(cfg.paths.document_images_dir)
    out = ROOT / "reports" / "metadata_audit"
    if not script.is_file():
        print(f"Audit script not found: {script}", file=sys.stderr)
        return 2
    cmd = [
        sys.executable,
        str(script),
        str(meta),
        "--image-dir",
        str(images),
        "--out",
        str(out),
        "--exclude",
        "*save*",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
