from __future__ import annotations
import os
from datetime import datetime
from typing import Iterable, Optional, Dict, Any

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv

def _get_client():
    load_dotenv(find_dotenv())
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI no definido (.env)")
    return MongoClient(uri, tz_aware=True)

def _get_collection(name: Optional[str] = None):
    client = _get_client()
    db = client[os.getenv("MONGO_DB")]
    coll = db[name or os.getenv("MONGO_COLL")]
    return coll

def ensure_indexes(name: Optional[str] = None):
    c = _get_collection(name)
    # índices útiles para rango temporal y sesión
    c.create_index([("Time", 1)], background=True)
    c.create_index([("NewSession", 1), ("Time", 1)], background=True)

def _build_projection(include_of: bool = False, extra_fields: Optional[Iterable[str]] = None) -> Dict[str, int]:
    base = ["Time","Open","High","Low","Close","MVC","Volume","Delta","NewSession","NewWeek","NewMonth"]
    if include_of:
        base += ["Ask","Bid"]
    if extra_fields:
        base += list(extra_fields)
    proj = {k: 1 for k in base}
    proj["_id"] = 0
    return proj

def read_klines(
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    coll_name: Optional[str] = None,
    include_of: bool = False,
    extra_fields: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    tz: str = "Europe/Madrid",
) -> pd.DataFrame:
    """
    Lee velas de Mongo por rango [start, end) y devuelve un DataFrame ordenado por Time.
    - include_of=True para incluir Ask/Bid (pesados).
    - tz: zona horaria del índice de salida (guardamos en UTC y convertimos al final).
    """
    c = _get_collection(coll_name)

    q: Dict[str, Any] = {}
    if start or end:
        q["Time"] = {}
        if start is not None:
            q["Time"]["$gte"] = pd.to_datetime(start, utc=True).to_pydatetime()
        if end is not None:
            q["Time"]["$lt"] = pd.to_datetime(end, utc=True).to_pydatetime()

    proj = _build_projection(include_of=include_of, extra_fields=extra_fields)

    cur = c.find(q, proj).sort("Time", 1).batch_size(5000)
    if limit:
        cur = cur.limit(int(limit))

    docs = list(cur)
    if not docs:
        return pd.DataFrame(columns=list(proj.keys()))

    df = pd.DataFrame(docs)
    # Asegurar datetime con tz:
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    if tz:
        df["Time"] = df["Time"].dt.tz_convert(tz)

    # Orden y tipos básicos
    df = df.sort_values("Time").reset_index(drop=True)
    for col in ["Open","High","Low","Close","MVC","Volume","Delta"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
