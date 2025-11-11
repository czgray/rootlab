"""I/O helpers for JSON and related artifacts."""
from pathlib import Path
import json
from typing import Any

def load_json(path: str | Path) -> Any:
    """Load a JSON file into a Python object."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Save a Python object to JSON."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
