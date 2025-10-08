# src/ppz/features/of_imbalance.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _row_imbalances(ask: np.ndarray, bid: np.ndarray, k: float, k_ext: float) -> Dict[str, float]:
    """
    Calcula imbalances diagonales por vela (desde High->Low):
      - buy if ask[i] >= k * bid[i+1]
      - sell if bid[i+1] >= k * ask[i]
    Devuelve contadores, intensidades, runs apilados (stacked) y localización (top/bottom thirds).
    """
    # asegura arrays 1D del mismo tamaño mínimo
    n = min(len(ask), len(bid))
    if n < 2:
        return dict(
            imb_buy_count=0, imb_sell_count=0,
            imb_buy_strength_avg=0.0, imb_sell_strength_avg=0.0,
            imb_extreme_flag=0, imb_buy_top_ratio=0.0, imb_sell_bottom_ratio=0.0,
            si_buy_count=0, si_sell_count=0, si_buy_max_len=0, si_sell_max_len=0
        )
    a = np.asarray(ask[:n], dtype=float)
    b = np.asarray(bid[:n], dtype=float)

    # comparaciones diagonales (i con i+1). Rango de índices válidos: 0..n-2
    A = a[:-1]
    B = b[1:]

    buy_mask  = A >= (k * B)
    sell_mask = B >= (k * a[:-1])

    # ratios (robustos)
    with np.errstate(divide='ignore', invalid='ignore'):
        buy_ratio  = np.where(buy_mask,  A / np.maximum(B, 1e-9), np.nan)
        sell_ratio = np.where(sell_mask, B / np.maximum(a[:-1], 1e-9), np.nan)

    buy_cnt  = int(np.nansum(buy_mask))
    sell_cnt = int(np.nansum(sell_mask))

    # medias robustas (nanmean) → sin warnings
    buy_strength_avg  = float(np.nanmean(buy_ratio))  if buy_cnt  > 0 else 0.0
    sell_strength_avg = float(np.nanmean(sell_ratio)) if sell_cnt > 0 else 0.0

    # extremos
    imb_extreme_flag = int(
        (np.nanmax(buy_ratio)  >= k_ext) if np.isfinite(np.nanmax(buy_ratio))  else False
        or
        (np.nanmax(sell_ratio) >= k_ext) if np.isfinite(np.nanmax(sell_ratio)) else False
    )

    # runs (stacked)
    def max_run(mask: np.ndarray):
        best = cur = runs = 0
        for v in mask:
            if v: cur += 1
            else:
                if cur>0: runs += 1
                best = max(best, cur); cur = 0
        if cur>0: runs += 1; best = max(best, cur)
        return runs, best

    si_buy_count, si_buy_max_len   = max_run(buy_mask.astype(bool))
    si_sell_count, si_sell_max_len = max_run(sell_mask.astype(bool))

    # terciles superior/inferior
    L = len(buy_mask)
    if L > 0:
        top_end = max(1, L//3)
        bot_start = L - top_end
        buy_top_ratio     = float(np.nanmean(buy_mask[:top_end])) if top_end>0 else 0.0
        sell_bottom_ratio = float(np.nanmean(sell_mask[bot_start:])) if bot_start<L else 0.0
    else:
        buy_top_ratio = sell_bottom_ratio = 0.0

    return dict(
        imb_buy_count=buy_cnt,
        imb_sell_count=sell_cnt,
        imb_buy_strength_avg=buy_strength_avg,
        imb_sell_strength_avg=sell_strength_avg,
        imb_extreme_flag=imb_extreme_flag,
        imb_buy_top_ratio=buy_top_ratio,
        imb_sell_bottom_ratio=sell_bottom_ratio,
        si_buy_count=int(si_buy_count),
        si_sell_count=int(si_sell_count),
        si_buy_max_len=int(si_buy_max_len),
        si_sell_max_len=int(si_sell_max_len),
    )


def compute_of_features_at(
    df: pd.DataFrame, idxs: np.ndarray, k: float = 3.0, k_ext: float = 5.0
) -> pd.DataFrame:
    """
    Calcula features de order flow SOLO en los índices 'idxs' (recomendado: indices de eventos).
    Requiere columnas: 'Ask' y 'Bid' como listas/arrays por vela.
    """
    assert {"Ask","Bid"}.issubset(df.columns), "Faltan columnas Ask/Bid"
    rows = []
    for i in idxs:
        ask = df.at[i, "Ask"]; bid = df.at[i, "Bid"]
        if ask is None or bid is None:
            feats = _row_imbalances(np.array([]), np.array([]), k, k_ext)
        else:
            feats = _row_imbalances(np.asarray(ask), np.asarray(bid), k, k_ext)
        feats["idx"] = int(i)
        rows.append(feats)
    out = pd.DataFrame(rows).set_index("idx").sort_index()
    # tipos compactos
    for c in out.columns:
        if out[c].dtype == float:
            out[c] = out[c].astype(np.float32)
        elif out[c].dtype == int:
            out[c] = out[c].astype(np.int16)
    return out.reset_index()
