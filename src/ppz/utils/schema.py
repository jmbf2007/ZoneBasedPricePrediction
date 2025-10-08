# src/ppz/utils/schema.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype


REQ_DF_BASE = [
    "Time","Open","High","Low","Close","Delta","Volume","MVC",
    "NewSession","NewWeek","NewMonth","Ask","Bid",
    # derivadas esperadas
    "session_id","idx_in_session","VWAP","session_tag",
]

REQ_EVENTS = [
    "idx","zone_type","level_price","dist_ticks",
    # aconsejadas
    "Time","session_id",
]

REQ_EVENTS_LABELED = REQ_EVENTS + ["label","session_tag"]

def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool: return s
    return s.fillna(False).astype(bool)

def _ensure_tz(s: pd.Series, tz: str) -> pd.Series:
    # 1) Fuerza a datetime si viene como object/str
    if not is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")

    # 2) Si ya es tz-aware, solo convierte
    if is_datetime64tz_dtype(s):
        return s.dt.tz_convert(tz)

    # 3) Si es naive, localiza y convierte
    #    Asumimos que los timestamps naive son UTC; cámbialo a "Europe/Madrid"
    #    si tus datos naive ya están en hora local.
    s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(tz)


def _compute_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "NewSession" not in out.columns:
        # si no existe, asume nueva sesión cuando Time pasa de un día a otro
        day = out["Time"].dt.date
        out["NewSession"] = day.ne(day.shift(1).fillna(day.iloc[0]))
    out["NewSession"] = _as_bool(out["NewSession"])
    out["session_id"] = out["NewSession"].cumsum().astype(int)

    # índice intradía
    out["idx_in_session"] = out.groupby("session_id").cumcount()
    return out

def _compute_vwap_by_session(df: pd.DataFrame) -> pd.Series:
    typ = (df["High"] + df["Low"] + df["Close"]) / 3.0
    ses = df.groupby("session_id", group_keys=False)
    cum_pv = ses.apply(lambda g: (typ.loc[g.index] * g["Volume"]).cumsum())
    cum_v  = ses["Volume"].cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap

def _session_tag_from_idx(idx_in_session: pd.Series,
                          bands: Tuple[Tuple[int,int,str], ...]) -> pd.Series:
    tag = pd.Series(index=idx_in_session.index, dtype="object")
    for lo, hi, name in bands:
        tag.loc[idx_in_session.between(lo, hi)] = name
    return tag.fillna("ASIA").astype("category")

def ensure_df_base_schema(
    df: pd.DataFrame,
    *,
    tz: str = "Europe/Madrid",
    idx_bands_5m: Tuple[Tuple[int,int,str], ...] = ((0,107,"ASIA"),(108,185,"EU"),(186,275,"USA")),
    compute_vwap: bool = True,
    placeholders_for_of: bool = True,
) -> pd.DataFrame:
    """
    Garantiza que df tenga las columnas mínimas y derivadas:
    Time(tz), session_id, idx_in_session, VWAP, session_tag, Ask/Bid si faltan.
    """
    out = df.copy()

    # Time tz-aware
    out["Time"] = _ensure_tz(out["Time"], tz)

    # Placeholders de orderflow si faltan (listas/arrays)
    if "Ask" not in out.columns and placeholders_for_of:
        out["Ask"] = [[] for _ in range(len(out))]
    if "Bid" not in out.columns and placeholders_for_of:
        out["Bid"] = [[] for _ in range(len(out))]

    # Bools semánticos
    for c in ["NewSession","NewWeek","NewMonth"]:
        if c in out.columns:
            out[c] = _as_bool(out[c])
        else:
            out[c] = False

    # Sesiones
    if "session_id" not in out.columns or "idx_in_session" not in out.columns:
        out = _compute_sessions(out)

    # VWAP
    if compute_vwap and "VWAP" not in out.columns:
        out["VWAP"] = _compute_vwap_by_session(out)

    # session_tag por bandas de índice intradía (5m → 276 velas)
    if "session_tag" not in out.columns:
        out["session_tag"] = _session_tag_from_idx(out["idx_in_session"], idx_bands_5m)

    # Tipos numéricos razonables
    num = ["Open","High","Low","Close","Delta","Volume","MVC","VWAP"]
    for c in num:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def ensure_events_schema(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    tick_size: float = 0.25,
    round_dist_ticks: bool = True,
) -> pd.DataFrame:
    """
    Asegura columnas mínimas en events: idx, zone_type, level_price, dist_ticks, Time, session_id.
    Rellena Time y session_id desde df via idx.
    """
    out = events.copy()

    # idx/zone_type obligatorios
    if "idx" not in out.columns:
        raise ValueError("events: falta columna 'idx'")
    if "zone_type" not in out.columns:
        raise ValueError("events: falta columna 'zone_type'")

    # level_price / dist_ticks
    if "level_price" not in out.columns:
        out["level_price"] = np.nan
    if "dist_ticks" not in out.columns:
        # si hay 'Close' o 'price' en events, aproxima; si no, NaN
        if "price" in out.columns:
            out["dist_ticks"] = (out["price"] - out["level_price"]) / tick_size
        else:
            out["dist_ticks"] = np.nan

    # map Time / session_id desde df
    time_map = df["Time"]
    sess_map = df["session_id"] if "session_id" in df.columns else None

    if "Time" not in out.columns or out["Time"].isna().any():
        out["Time"] = out["idx"].map(time_map)

    if "session_id" not in out.columns or out["session_id"].isna().any():
        if sess_map is not None:
            out["session_id"] = out["idx"].map(sess_map)

    # limpieza de dist_ticks
    out["dist_ticks"] = pd.to_numeric(out["dist_ticks"], errors="coerce")
    if round_dist_ticks:
        out["dist_ticks"] = np.round(out["dist_ticks"]).astype("Int64")  # entero tolerante a NaN

    return out

def ensure_events_labeled_schema(
    df: pd.DataFrame,
    events_labeled: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extiende ensure_events_schema() añadiendo 'label' y 'session_tag' consistentes.
    """
    base = ensure_events_schema(df, events_labeled)
    out = base.copy()

    # label
    if "label" not in out.columns:
        out["label"] = "none"
    out["label"] = out["label"].astype(str)

    # session_tag desde df por idx (si falta)
    if "session_tag" not in out.columns or out["session_tag"].isna().any():
        if "session_tag" in df.columns:
            out["session_tag"] = out["idx"].map(df["session_tag"]).astype("category")
        else:
            out["session_tag"] = "ASIA"

    return out

def assert_has_columns(df: pd.DataFrame, cols: Iterable[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{name} sin columnas requeridas: {missing}")
