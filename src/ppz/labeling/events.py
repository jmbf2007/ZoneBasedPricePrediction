# src/ppz/labeling/events.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

__all__ = ["LabelParams", "label_events", "join_labels_with_events"]

@dataclass
class LabelParams:
    tick_size: float = 0.25
    H_horizon_bars: int = 12      # horizonte para evaluar (H)
    p_inval_ticks: int = 6        # penetración que invalida la zona (p)
    X_rebound_ticks: int = 10     # avance mínimo para considerar rebote (X)
    Y_break_confirm_ticks: int = 16  # confirmación de ruptura (Y)
    restrict_same_session: bool = True  # no cruzar la sesión en la ventana H

def _first_hit_idx(high: np.ndarray, low: np.ndarray, thresh: float, side: str) -> Optional[int]:
    """
    Devuelve el primer índice (1..N) donde se cruza el umbral.
    side='up' usa high>=thresh; side='down' usa low<=thresh.
    """
    if side == "up":
        hit = np.argmax(high >= thresh)
        return int(hit)+1 if high.size and (high >= thresh).any() else None
    else:
        hit = np.argmax(low <= thresh)
        return int(hit)+1 if low.size and (low <= thresh).any() else None

def _slice_window(df: pd.DataFrame, i: int, H: int, restrict_session: bool) -> pd.DataFrame:
    """ Extrae df[i+1 : i+H] y (opcional) corta al cambiar de sesión. """
    end = min(i + H, len(df) - 1)
    w = df.iloc[i+1 : end+1]
    if restrict_session and not w.empty:
        sid = df.iloc[i]["session_id"]
        w = w[w["session_id"] == sid]
    return w

def _label_one(df: pd.DataFrame, i: int, level: float, params: LabelParams) -> Dict[str, Any]:
    """
    Etiqueta un evento en índice global i contra 'level'.
    Regla:
      - side_at_event = 'resistance' si Close_i < level; 'support' si Close_i > level; empate → decide por cuerpo (si Close==level, usa Open).
      - REBOTE: alcanza X ticks al lado opuesto ANTES de invalidar (p).
      - RUPTURA: invalida (p) y confirma Y ticks en la misma dirección ANTES de un rebote previo.
      - NONE si ninguna condición ocurre en H barras.
    """
    px0 = float(df.iloc[i]["Close"])
    op0 = float(df.iloc[i]["Open"])
    side = "resistance" if px0 < level else ("support" if px0 > level else ("resistance" if op0 < level else "support"))

    H = int(params.H_horizon_bars)
    tw = _slice_window(df, i, H, params.restrict_same_session)
    if tw.empty:
        return {
            "idx": int(i), "label": "none", "side_at_event": side,
            "t_rebound": None, "t_rupture": None,
            "mfe_up_ticks": 0.0, "mfe_down_ticks": 0.0
        }

    high = tw["High"].to_numpy(dtype=float)
    low  = tw["Low"].to_numpy(dtype=float)

    tick = float(params.tick_size)
    p = tick * params.p_inval_ticks
    X = tick * params.X_rebound_ticks
    Y = tick * params.Y_break_confirm_ticks

    # Umbrales
    up_inval   = level + p
    down_inval = level - p
    up_reb     = level + X
    down_reb   = level - X
    up_confY   = level + Y
    down_confY = level - Y

    # Tiempos de hit
    t_up_inval   = _first_hit_idx(high, low, up_inval, "up")
    t_down_inval = _first_hit_idx(high, low, down_inval, "down")
    t_up_reb     = _first_hit_idx(high, low, up_reb, "up")
    t_down_reb   = _first_hit_idx(high, low, down_reb, "down")
    t_up_confY   = _first_hit_idx(high, low, up_confY, "up")
    t_down_confY = _first_hit_idx(high, low, down_confY, "down")

    # MFE relativos al nivel en ambas direcciones
    mfe_up_ticks = max(0.0, (np.max(high) - level) / tick) if high.size else 0.0
    mfe_down_ticks = max(0.0, (level - np.min(low)) / tick) if low.size else 0.0

    # Decisión
    label = "none"
    t_reb = None
    t_rup = None

    if side == "resistance":
        # Rebote hacia abajo si toca down_reb antes de invalidar por arriba
        if t_down_reb is not None and (t_up_inval is None or t_down_reb < t_up_inval):
            label, t_reb = "rebound", t_down_reb
        # Ruptura al alza si invalida y confirma Y (y no hubo rebote anterior)
        elif t_up_inval is not None and t_up_confY is not None and t_up_inval <= t_up_confY and (t_down_reb is None or t_up_confY <= t_down_reb):
            label, t_rup = "breakout", t_up_confY
    else:  # support
        # Rebote hacia arriba si toca up_reb antes de invalidar por abajo
        if t_up_reb is not None and (t_down_inval is None or t_up_reb < t_down_inval):
            label, t_reb = "rebound", t_up_reb
        # Ruptura a la baja si invalida y confirma Y (y no hubo rebote anterior)
        elif t_down_inval is not None and t_down_confY is not None and t_down_inval <= t_down_confY and (t_up_reb is None or t_down_confY <= t_up_reb):
            label, t_rup = "breakout", t_down_confY

    return {
        "idx": int(i),
        "label": label,
        "side_at_event": side,
        "t_rebound": int(t_reb) if t_reb is not None else None,
        "t_rupture": int(t_rup) if t_rup is not None else None,
        "mfe_up_ticks": float(mfe_up_ticks),
        "mfe_down_ticks": float(mfe_down_ticks),
    }

def label_events(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    params: Optional[LabelParams] = None,
    level_col: str = "level_price",
) -> pd.DataFrame:
    """
    Etiqueta todos los eventos:
    devuelve DataFrame con columnas:
      ['idx','label','side_at_event','t_rebound','t_rupture','mfe_up_ticks','mfe_down_ticks']
    """
    if params is None:
        params = LabelParams()
    assert "session_id" in df.columns and "High" in df.columns and "Low" in df.columns and "Open" in df.columns and "Close" in df.columns

    out_rows: List[Dict[str, Any]] = []
    # Si no tenemos level_price en events_df, lo reconstruimos desde df[zone_type] si existe esa col
    need_level = level_col not in events_df.columns
    for _, r in events_df.iterrows():
        i = int(r["idx"])
        level = float(r[level_col]) if not need_level else float(df.loc[i, str(r["zone_type"])])
        out_rows.append(_label_one(df, i, level, params))

    return pd.DataFrame(out_rows)

def join_labels_with_events(events_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """ Merge por 'idx' conservando columnas del events_df. """
    return events_df.merge(labels_df, on="idx", how="left")
