# src/ppz/features/zones.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Dict
import numpy as np
import pandas as pd

__all__ = [
    "add_session_indices",
    "compute_atr",
    "compute_pdh_pdl_prev",
    "compute_usa_ib",
    "compute_vwap_and_bands",
    "detect_events_to_level",
    "build_event_table",
]

# ---------------------------
# Session, ATR, daily levels
# ---------------------------

def add_session_indices(
    df: pd.DataFrame,
    new_session_col: str = "NewSession",
    session_col: str = "session_id",
    pos_col: str = "idx_in_session",
) -> pd.DataFrame:
    """
    Añade identificador de sesión y posición intradía basada en `new_session_col`.
    No modifica el orden; asume df ya ordenado por Time.
    """
    out = df.copy()
    out[session_col] = (out[new_session_col] == True).cumsum()
    out[pos_col] = out.groupby(session_col).cumcount()
    return out



def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    out_col: str = "ATR_14",
) -> pd.DataFrame:
    """
    ATR(14) clásico usando EWM (tipo Wilder aprox).
    """
    out = df.copy()
    high, low, close = out["High"], out["Low"], out["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    # EWM con alpha=1/period ≈ Wilder
    out[out_col] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return out


def compute_pdh_pdl_prev(
    df: pd.DataFrame,
    session_col: str = "session_id",
    out_pdh: str = "PDH_prev",
    out_pdl: str = "PDL_prev",
) -> pd.DataFrame:
    """
    PDH/PDL del DÍA PREVIO por sesión (usa máximos/mínimos de la sesión anterior).
    """
    out = df.copy()
    ses = out.groupby(session_col)
    day_hi_prev = ses["High"].transform("max").groupby(out[session_col]).shift(1)
    day_lo_prev = ses["Low"].transform("min").groupby(out[session_col]).shift(1)

    # Mapear por sesión evita errores si hay gaps
    hi_map = ses["High"].max().shift(1)
    lo_map = ses["Low"].min().shift(1)
    out[out_pdh] = out[session_col].map(hi_map)
    out[out_pdl] = out[session_col].map(lo_map)
    return out


# ---------------------------
# USA IB detection
# ---------------------------

def _detect_usa_open_idx_per_session(
    g: pd.DataFrame,
    idx_col: str = "idx_in_session",
    search: Tuple[int, int] = (120, 220),
    vol_weight: float = 0.6,
    tr_weight: float = 0.4,
) -> int:
    """
    Devuelve el índice intradía (idx_in_session) estimado de apertura USA dentro de una ventana de búsqueda.
    Heurística: maximiza score = w*Z(Volume) + (1-w)*Z(TrueRange).
    """
    i0, i1 = search
    gi = g[(g[idx_col] >= i0) & (g[idx_col] <= i1)].copy()
    if gi.empty:
        return -1
    # Z-scores simples
    zvol = (gi["Volume"] - gi["Volume"].mean()) / (gi["Volume"].std(ddof=0) + 1e-12)
    tr = (gi["High"] - gi["Low"]).abs()
    ztr = (tr - tr.mean()) / (tr.std(ddof=0) + 1e-12)
    score = vol_weight * zvol + tr_weight * ztr
    return int(gi.loc[score.idxmax(), idx_col])


def compute_usa_ib(
    df: pd.DataFrame,
    session_col: str = "session_id",
    idx_col: str = "idx_in_session",
    ib_bars: int = 12,                      # 1h en 5m
    fixed_open_idx: Optional[int] = 186,    # ← fijamos 15:30 Madrid
    search: Tuple[int, int] = (120, 220),   # (solo si fixed_open_idx=None)
    vol_weight: float = 0.6,
    tr_weight: float = 0.4,
    out_open_col: str = "usa_open_idx",
    out_ibh: str = "USA_IBH",
    out_ibl: str = "USA_IBL",
) -> pd.DataFrame:
    """
    Si `fixed_open_idx` no es None, usa ese índice como apertura USA en TODAS las sesiones.
    Si es None, aplica la detección heurística dentro de `search`.
    Devuelve columnas: usa_open_idx, USA_IBH, USA_IBL.
    """
    out = df.copy()

    # Asegura columnas base
    if session_col not in out or idx_col not in out:
        raise ValueError(f"Faltan columnas {session_col} o {idx_col}. Ejecuta add_session_indices() antes.")

    if fixed_open_idx is not None:
        # Fijamos el open USA idéntico para todas las sesiones
        out[out_open_col] = int(fixed_open_idx)
    else:
        # Detección por zscores (fallback)
        ses = out.groupby(session_col)
        open_idx = ses.apply(
            _detect_usa_open_idx_per_session,
            idx_col=idx_col, search=search,
            vol_weight=vol_weight, tr_weight=tr_weight
        )
        # asegura nombre correcto y merge robusto
        open_idx = open_idx.rename(out_open_col).to_frame()
        out = out.merge(open_idx, left_on=session_col, right_index=True, how="left")

    # Calcula IB por sesión usando usa_open_idx
    ibh = pd.Series(index=out.index, dtype=float)
    ibl = pd.Series(index=out.index, dtype=float)

    for sid, g in out.groupby(session_col):
        # si por lo que sea no está la col, default a fixed_open_idx
        oi = int(g[out_open_col].iloc[0]) if out_open_col in g.columns else int(fixed_open_idx or -1)
        if oi < 0:
            ibh_val = np.nan
            ibl_val = np.nan
        else:
            start = oi
            stop  = oi + ib_bars
            win = g[(g[idx_col] >= start) & (g[idx_col] < stop)]
            if win.empty:
                ibh_val = np.nan
                ibl_val = np.nan
            else:
                ibh_val = float(win["High"].max())
                ibl_val = float(win["Low"].min())
        ibh.loc[g.index] = ibh_val
        ibl.loc[g.index] = ibl_val

    out[out_ibh] = ibh
    out[out_ibl] = ibl
    return out



# ---------------------------
# VWAP and bands
# ---------------------------

# src/ppz/features/zones.py  (reemplaza la función compute_vwap_and_bands por esta)

def compute_vwap_and_bands(
    df: pd.DataFrame,
    session_col: str = "session_id",
    sigma: float = 1.0,
    out_vwap: str = "VWAP",
    out_std: str = "VWAP_std",
    out_p1s: str = "VWAP_p1s",
    out_m1s: str = "VWAP_m1s",
) -> pd.DataFrame:
    """
    VWAP intradía acumulado por sesión y bandas ±sigma con std acumulada
    de (Close - VWAP) dentro de la sesión (usando sumas acumuladas para estabilidad).
    """
    out = df.copy()

    # VWAP = cumsum(P*V) / cumsum(V) por sesión (índices alineados)
    typ = (out["High"] + out["Low"] + out["Close"]) / 3.0
    pv = typ * out["Volume"]
    cum_pv = pv.groupby(out[session_col]).cumsum()
    cum_v  = out["Volume"].groupby(out[session_col]).cumsum()
    # evita división por 0 al principio de la sesión
    out[out_vwap] = (cum_pv / cum_v.replace(0, np.nan))

    # Desviación estándar acumulada de dev = Close - VWAP por sesión:
    # var = E[x^2] - (E[x])^2 con E acumulada
    dev = out["Close"] - out[out_vwap]
    grp = out[session_col]
    n   = grp.groupby(grp).cumcount() + 1  # 1,2,3,... dentro de cada sesión
    sum1 = dev.groupby(grp).cumsum()
    sum2 = (dev**2).groupby(grp).cumsum()

    mean = sum1 / n
    var  = (sum2 / n) - (mean**2)
    var  = np.clip(var, 0.0, None)  # evita negativos por redondeo
    std  = np.sqrt(var)

    out[out_std] = std
    out[out_p1s] = out[out_vwap] + sigma * std
    out[out_m1s] = out[out_vwap] - sigma * std
    return out



# ----------------------------------------
# Detector de eventos (mejorado: CROSSING)
# ----------------------------------------

def detect_events_to_level(
    price: pd.Series,
    level: pd.Series,
    r_ticks: int = 6,
    tick_size: float = 0.25,
    cooldown: int = 10,
    mode: str = "cross",               # "cross" | "touch"
) -> np.ndarray:
    """
    Devuelve índices donde el precio llega a un nivel.
    - 'touch': |price-level| <= r.
    - 'cross': entra DESDE FUERA del radio (near_t & ~near_{t-1}).
    Se aplica 'cooldown' en velas para no repetir.
    """
    level = level.astype(float)
    valid = ~level.isna()
    if not valid.any():
        return np.array([], dtype=int)

    dist_ticks = (price[valid] - level[valid]).abs() / float(tick_size)
    near = dist_ticks <= float(r_ticks)
    near_prev = near.shift(1, fill_value=False)

    if mode == "cross":
        trig = near & (~near_prev)
    else:  # "touch"
        trig = near.astype(bool)

    idxs_all = price.index[valid].to_numpy()
    idxs: List[int] = []
    last = -10**9
    for local_i, flag in enumerate(trig):
        if bool(flag):
            i = int(idxs_all[local_i])
            if i - last > cooldown:
                idxs.append(i)
                last = i
    return np.array(idxs, dtype=int)



@dataclass
class EventSpec:
    """Especificación de las zonas a chequear para eventos."""
    column: str     # nombre de columna con el nivel (ej. 'PDH_prev')
    alias: str      # etiqueta en la tabla de eventos (ej. 'PDH_prev')


def build_event_table(
    df: pd.DataFrame,
    zones: Iterable[EventSpec],
    session_col: str = "session_id",
    time_col: str = "Time",
    price_col: str = "Close",
    r_prox: int = 6,
    cooldown: int = 10,
    tick_size: float = 0.25,
    mode: str = "cross",                      # "cross" | "touch"
    dedupe: str = "priority",                 # "none" | "priority"
    zone_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Genera tabla de eventos de llegada con columnas:
    ['idx','session_id','Time','zone_type','level_price','dist_ticks'].
    - idx es el ÍNDICE GLOBAL del df.
    """
    records: List[Tuple[int,int,pd.Timestamp,str,float,float]] = []

    # Itera por sesión y zona
    for sid, g in df.groupby(session_col):
        for z in zones:
            # skip si la columna no existe o todo NaN en esta sesión
            if z.column not in df.columns:
                continue
            ser_level = g[z.column].astype(float)
            if not ser_level.notna().any():
                continue

            # Detecta llegadas (devuelve índices del df, no posiciones)
            idx_local = detect_events_to_level(
                g[price_col], ser_level,
                r_ticks=r_prox, tick_size=tick_size,
                cooldown=cooldown, mode=mode
            )
            if idx_local.size == 0:
                continue

            # Añade registros usando índice GLOBAL gi
            for gi in idx_local:
                gi = int(gi)
                lvl = float(df.at[gi, z.column])
                px  = float(df.at[gi, price_col])
                # distancia en ticks (redondeo half-up a entero)
                dist_raw   = abs(px - lvl) / float(tick_size)
                dist_ticks = int(np.floor(dist_raw + 0.5))   # entero
                records.append((gi, int(sid), df.at[gi, time_col], z.alias, lvl, dist_ticks))

    # Si no hubo eventos, devuelve DF vacío con esquema correcto
    if not records:
        return pd.DataFrame(columns=["idx","session_id","Time","zone_type","level_price","dist_ticks"])

    ev = (pd.DataFrame(records, columns=["idx","session_id","Time","zone_type","level_price","dist_ticks"])
            .sort_values(["idx","zone_type"])
            .reset_index(drop=True))

    # Dedupe opcional por prioridad de zona
    if dedupe == "priority":
        w = zone_weights or {}
        ev["z_weight"] = ev["zone_type"].map(lambda z: float(w.get(z, 1.0)))
        ev = (ev.sort_values(["idx","z_weight","dist_ticks"], ascending=[True, False, True])
                .groupby("idx", as_index=False).head(1)
                .sort_values("idx")
                .reset_index(drop=True)
              ).drop(columns=["z_weight"])

    return ev
