# src/ppz/io/paths.py
from pathlib import Path
def root() -> Path: return Path(__file__).resolve().parents[3]
def ensure(p: Path): p.parent.mkdir(parents=True, exist_ok=True)
def interim(name:str)->Path: p=root()/ "data"/"interim"/name; ensure(p); return p
def events(name:str)->Path:  p=root()/ "data"/"processed"/"events"/name; ensure(p); return p
def feats(name:str)->Path:   p=root()/ "data"/"processed"/"features"/name; ensure(p); return p
