"""UTF-8 I/O helpers and JSON Schema validation.

Windows default locale is often cp1252; every file read/write in this project
must go through these helpers (or an explicit encoding=\"utf-8\" call).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

PathLike = Union[str, Path]

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
_SCHEMA_CACHE: dict[str, Draft202012Validator] = {}


def read_text(path: PathLike) -> str:
    """Read a text file as UTF-8 (accept UTF-8 BOM via utf-8-sig)."""
    p = Path(path)
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return p.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


def write_text(path: PathLike, text: str) -> None:
    """Write text as UTF-8, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: PathLike) -> Any:
    """Load a JSON file as UTF-8."""
    return json.loads(read_text(path))


def write_json(path: PathLike, data: Any, *, indent: int = 2) -> None:
    """Dump JSON as UTF-8 with a trailing newline."""
    text = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    write_text(path, text)


def write_csv(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Optional[list[str]] = None,
) -> None:
    """Write a CSV with UTF-8 encoding and newline=\"\" (Windows-safe)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("write_csv requires fieldnames when rows is empty")
        fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def schema_path(name: str) -> Path:
    """Resolve a schema file under schemas/ (name with or without .json)."""
    fname = name if name.endswith(".json") else f"{name}.json"
    path = _SCHEMA_DIR / fname
    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {path}")
    return path


def load_schema(name: str) -> dict[str, Any]:
    """Load a JSON Schema document by short name (e.g. 'segment')."""
    return read_json(schema_path(name))


def get_validator(name: str) -> Draft202012Validator:
    """Return a cached Draft 2020-12 validator for the named schema."""
    key = name if name.endswith(".json") else f"{name}.json"
    if key not in _SCHEMA_CACHE:
        schema = load_schema(key)
        _SCHEMA_CACHE[key] = Draft202012Validator(schema)
    return _SCHEMA_CACHE[key]


def validate(instance: Any, schema_name: str) -> None:
    """Validate *instance* against *schema_name*; raise ValidationError on failure."""
    get_validator(schema_name).validate(instance)


def is_valid(instance: Any, schema_name: str) -> bool:
    """Return True if *instance* validates against *schema_name*."""
    return get_validator(schema_name).is_valid(instance)


def validation_errors(instance: Any, schema_name: str) -> list[ValidationError]:
    """Return all validation errors for *instance* against *schema_name*."""
    return list(get_validator(schema_name).iter_errors(instance))
