# src/ppz/io/paths.py
from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # .../src/ppz/io/paths.py -> repo root
    return Path(__file__).resolve().parents[3]

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def interim_path(name: str) -> Path:
    p = project_root() / "data" / "interim" / name
    ensure_dir(p); return p

def events_path(name: str) -> Path:
    p = project_root() / "data" / "processed" / "events" / name
    ensure_dir(p); return p
