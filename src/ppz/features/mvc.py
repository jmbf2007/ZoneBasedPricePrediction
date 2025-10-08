# src/ppz/features/mvc.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence

EPS = 1e-9

@dataclass
class MvcFlagsParams:
    tick_size: float = 0.25
    mvc_lower: float = 1/3          # MVC “abajo” si pos ≤ mvc_lower
    mvc_upper: float = 2/3          # MVC “arriba” si pos ≥ mvc_upper
    vwap_slope_window: int = 20
    vwap_flat_th_ticks_per_bar: float = 0.05
    first_touch_static: bool = True

def _bar_parts(o: float, h: float, l: float, c: float, eps: float):
    rng = max(h - l, eps)
    body = abs(c - o) / rng
    upper = (h - max(o, c)) / rng
    lower = (min(o, c) - l) / rng
    bull = c >= o
    return rng, body, upper, lower, bull

def _safe_div(a, b):
    return np.divide(a, np.where(np.abs(b) < EPS, np.nan, b))
def compute_mvc_features(
    df: pd.DataFrame,
    idx: Optional[Sequence[int]] = None,
    *,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """
    Devuelve features MVC por vela. Si 'idx' es None, calcula para todas.
    Si 'idx' es array-like, calcula SOLO para esas posiciones y devuelve en ese orden.

    Columns devueltas (prefijo mvc_):
      - mvc_pos            ∈ [0,1]  (posición relativa dentro del rango de la vela)
      - mvc_offset_ticks   (MVC - centro del cuerpo) / tick_size
      - mvc_to_close_ticks (Close - MVC) / tick_size  (signado)
      - body_ratio         |Close-Open| / (High-Low)
      - dir                {-1,0,1}  (bajista, doji, alcista)
      - upper_wick_ratio   (High - max(Open,Close)) / (High-Low)
      - lower_wick_ratio   (min(Open,Close) - Low) / (High-Low)
      - near_top_flag      1 si mvc_pos >= 0.66
      - near_bottom_flag   1 si mvc_pos <= 0.34
    """
    if idx is None:
        idx_arr = np.arange(len(df), dtype=int)
    else:
        idx_arr = np.asarray(idx, dtype=int).ravel()

    sub = df.iloc[idx_arr]

    o = sub["Open"].to_numpy(dtype=float)
    h = sub["High"].to_numpy(dtype=float)
    l = sub["Low"].to_numpy(dtype=float)
    c = sub["Close"].to_numpy(dtype=float)
    m = sub["MVC"].to_numpy(dtype=float)

    rng   = h - l
    body  = np.abs(c - o)
    ctr   = 0.5 * (o + c)

    mvc_pos = _safe_div(m - l, rng)  # [0..1] idealmente
    mvc_pos = np.clip(mvc_pos, 0.0, 1.0)

    mvc_offset_ticks   = _safe_div(m - ctr, tick_size)
    mvc_to_close_ticks = _safe_div(c - m, tick_size)  # signado

    body_ratio       = _safe_div(body, rng)
    upper_wick_ratio = _safe_div(h - np.maximum(o, c), rng)
    lower_wick_ratio = _safe_div(np.minimum(o, c) - l, rng)

    dir_ = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)

    near_top_flag    = (mvc_pos >= 0.66).astype(np.int8)
    near_bottom_flag = (mvc_pos <= 0.34).astype(np.int8)

    out = pd.DataFrame({
        "mvc_pos": mvc_pos.astype(np.float32),
        "mvc_offset_ticks": mvc_offset_ticks.astype(np.float32),
        "mvc_to_close_ticks": mvc_to_close_ticks.astype(np.float32),
        "body_ratio": np.nan_to_num(body_ratio, nan=0.0).astype(np.float32),
        "dir": dir_,
        "upper_wick_ratio": np.nan_to_num(upper_wick_ratio, nan=0.0).astype(np.float32),
        "lower_wick_ratio": np.nan_to_num(lower_wick_ratio, nan=0.0).astype(np.float32),
        "near_top_flag": near_top_flag,
        "near_bottom_flag": near_bottom_flag,
    }, index=idx_arr)

    # prefijo estandarizado
    out = out.add_prefix("mvc_")

    # mantiene el ORDEN de entrada
    out = out.loc[idx_arr].reset_index(drop=True)
    return out

def _classify_vwap_slope(vwap: pd.Series, window: int, tick_size: float, flat_th: float) -> pd.Series:
    # pendiente media por barra (ticks/bar)
    dv = vwap.diff(window)
    slope_per_bar = dv / max(window, 1)
    slope_in_ticks = slope_per_bar / tick_size
    cls = pd.Series(index=vwap.index, dtype="object")
    cls.loc[slope_in_ticks >  flat_th] = "up"
    cls.loc[slope_in_ticks < -flat_th] = "down"
    cls.fillna("flat", inplace=True)
    return cls

def annotate_mvc_directional_flags(df: pd.DataFrame, events: pd.DataFrame, params: MvcFlagsParams) -> pd.DataFrame:
    ev = events.copy()

    # Primer toque solo en zonas estáticas (opcional)
    if params.first_touch_static:
        static = {"PDH_prev","PDL_prev","USA_IBH","USA_IBL","VAH_D1","POC_D1","VAL_D1"}
        mask = ev["zone_type"].isin(static)
        ev_static = (ev[mask]
                     .sort_values(["session_id","zone_type","idx"])
                     .groupby(["session_id","zone_type"], as_index=False)
                     .nth(0))
        ev = pd.concat([ev_static, ev[~mask]], ignore_index=True).sort_values("idx").reset_index(drop=True)

    # Features/flags MVC por idx
    feats = [compute_mvc_features(
                df, i,
                tick_size=params.tick_size,
                mvc_lower=params.mvc_lower,
                mvc_upper=params.mvc_upper
            )
            for i in ev["idx"].astype(int)]
    F = pd.DataFrame(feats).reset_index(drop=True)
    out = pd.concat([ev.reset_index(drop=True), F], axis=1)

    # Lado de la zona (support/resistance)
    sup_types = {"USA_IBL","PDL_prev","VAL_D1"}
    res_types = {"USA_IBH","PDH_prev","VAH_D1"}

    side = []
    for i, z in zip(out["idx"].astype(int), out["zone_type"].astype(str)):
        if z in sup_types:
            side.append("support")
        elif z in res_types:
            side.append("resistance")
        elif z.startswith("VWAP"):
            # dinámico segun posición del precio respecto a VWAP
            if "VWAP" in df.columns:
                px = float(df.iloc[i]["Close"])
                lvl = float(df.iloc[i]["VWAP"])
                side.append("support" if px >= lvl else "resistance")
            else:
                side.append("neutral")
        else:
            side.append("neutral")
    out["side"] = pd.Series(side, dtype="category")

    # Clasificación de pendiente VWAP
    if "VWAP" in df.columns:
        cls = _classify_vwap_slope(
            df["VWAP"].astype(float),
            params.vwap_slope_window,
            params.tick_size,
            params.vwap_flat_th_ticks_per_bar
        )
        out["vwap_slope_class"] = cls.loc[out["idx"].astype(int)].to_numpy()
    else:
        out["vwap_slope_class"] = "flat"

    return out


