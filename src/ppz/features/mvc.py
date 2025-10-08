# src/ppz/features/mvc.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

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

def compute_mvc_features(
    df: pd.DataFrame,
    idx: int,
    tick_size: float = 0.25,
    mvc_lower: float = 1/3,
    mvc_upper: float = 2/3,
) -> dict:
    """
    Devuelve features/flags de MVC para la vela en 'idx'.
    - mvc_pos ∈ [0,1] relativo al rango [Low,High]
    - body/upper/lower wick ratios
    - flags direccionales: exh_up/exh_down, reject_up/reject_down
      · exhaustion: MVC en extremo (no imponemos body grande)
      · reject: MVC en extremo opuesto al color del cuerpo
    """
    row = df.iloc[int(idx)]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    mvc = float(row.get("MVC", np.nan))
    eps = tick_size

    rng, body, upper, lower, bull = _bar_parts(o, h, l, c, eps)

    if np.isnan(mvc) or rng <= 0:
        mvc_pos = np.nan
    else:
        mvc_pos = (mvc - l) / max(rng, eps)
        mvc_pos = float(np.clip(mvc_pos, 0.0, 1.0))

    exh_up   = int((not np.isnan(mvc_pos)) and (mvc_pos >= mvc_upper))
    exh_down = int((not np.isnan(mvc_pos)) and (mvc_pos <= mvc_lower))
    reject_up   = int((not np.isnan(mvc_pos)) and (not bull) and (mvc_pos >= mvc_upper))
    reject_down = int((not np.isnan(mvc_pos)) and (bull) and (mvc_pos <= mvc_lower))

    return {
        "mvc_pos": mvc_pos,
        "body_ratio": float(body),
        "upper_wick_ratio": float(upper),
        "lower_wick_ratio": float(lower),
        "exh_up": np.int8(exh_up),
        "exh_down": np.int8(exh_down),
        "reject_up": np.int8(reject_up),
        "reject_down": np.int8(reject_down),
    }

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


