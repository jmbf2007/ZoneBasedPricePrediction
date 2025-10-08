# src/ppz/features/vp.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

__all__ = ["compute_vah_poc_val_prev"]

def _build_price_grid(row: pd.Series, tick: float) -> np.ndarray:
    # grid High→Low inclusive; longitud = (High-Low)/tick + 1
    n = int(round((row["High"] - row["Low"]) / tick)) + 1
    # evitar n <= 0 por datos raros
    n = max(n, 1)
    return np.round(np.linspace(row["High"], row["Low"], n), 10)

def _session_profile(g: pd.DataFrame, tick: float) -> Dict[float, float]:
    """
    Acumula volumen por precio en la sesión: sum(Ask+Bid) para cada nivel.
    Requiere columnas Ask y Bid como listas alineadas con el grid de la vela.
    """
    vol_by_price: Dict[float, float] = {}
    for _, r in g.iterrows():
        if not isinstance(r.get("Ask"), (list, tuple)) or not isinstance(r.get("Bid"), (list, tuple)):
            continue
        ask = np.asarray(r["Ask"], dtype=float)
        bid = np.asarray(r["Bid"], dtype=float)
        if ask.size != bid.size or ask.size == 0:
            continue
        grid = _build_price_grid(r, tick)
        if grid.size != ask.size:  # sanity check
            continue
        vol = ask + bid
        # acumular
        for p, v in zip(grid, vol):
            vol_by_price[p] = vol_by_price.get(float(p), 0.0) + float(v)
    return vol_by_price

def _poc_vah_val_from_profile(vol_by_price: Dict[float, float], value_area: float = 0.70) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Devuelve (VAH, POC, VAL) siguiendo el criterio estándar:
    - POC = precio con mayor volumen
    - Value Area = ~70% del volumen total alrededor del POC, añadiendo precios
      adyacentes por volumen descendente (alternando lados) hasta cubrir el %.
    """
    if not vol_by_price:
        return (None, None, None)

    # ordenar niveles de precio (asc) y localizar POC
    prices = np.array(sorted(vol_by_price.keys()))
    vols   = np.array([vol_by_price[p] for p in prices], dtype=float)
    if vols.sum() <= 0:
        return (None, None, None)

    poc_idx = int(vols.argmax())
    poc_price = float(prices[poc_idx])

    total = float(vols.sum())
    target = value_area * total

    # expandir alrededor del POC
    included = {poc_idx}
    cum = float(vols[poc_idx])
    left = poc_idx - 1
    right = poc_idx + 1

    while cum < target and (left >= 0 or right < len(prices)):
        v_left = vols[left] if left >= 0 else -1.0
        v_right = vols[right] if right < len(prices) else -1.0
        if v_left >= v_right:
            if left >= 0:
                included.add(left)
                cum += float(vols[left])
                left -= 1
            else:
                if right < len(prices):
                    included.add(right)
                    cum += float(vols[right])
                    right += 1
                else:
                    break
        else:
            if right < len(prices):
                included.add(right)
                cum += float(vols[right])
                right += 1
            else:
                if left >= 0:
                    included.add(left)
                    cum += float(vols[left])
                    left -= 1
                else:
                    break

    vah = float(prices[max(included)])
    val = float(prices[min(included)])
    return (vah, poc_price, val)

def compute_vah_poc_val_prev(
    df: pd.DataFrame,
    tick_size: float = 0.25,
    session_col: str = "session_id",
    out_vah: str = "VAH_D1",
    out_poc: str = "POC_D1",
    out_val: str = "VAL_D1",
    value_area: float = 0.70,
) -> pd.DataFrame:
    """
    Construye VAH/POC/VAL por sesión a partir del profile (Ask+Bid) y los
    asigna a la **siguiente** sesión (D-1 respecto a la actual).
    """
    out = df.copy()
    sess_levels: Dict[int, Tuple[Optional[float], Optional[float], Optional[float]]] = {}

    for sid, g in out.groupby(session_col):
        prof = _session_profile(g, tick=tick_size)
        vah, poc, val = _poc_vah_val_from_profile(prof, value_area=value_area)
        sess_levels[int(sid)] = (vah, poc, val)

    # mapear niveles de la sesión anterior → esta sesión
    vah_map = {sid+1: sess_levels[sid][0] for sid in sess_levels if sess_levels[sid] is not None}
    poc_map = {sid+1: sess_levels[sid][1] for sid in sess_levels if sess_levels[sid] is not None}
    val_map = {sid+1: sess_levels[sid][2] for sid in sess_levels if sess_levels[sid] is not None}

    out[out_vah] = out[session_col].map(vah_map)
    out[out_poc] = out[session_col].map(poc_map)
    out[out_val] = out[session_col].map(val_map)
    return out
