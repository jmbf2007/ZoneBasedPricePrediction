# src/ppz/pipelines/build_dataset.py
from __future__ import annotations
from typing import Iterable,Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

from ppz.features.mvc import compute_mvc_features
from ppz.features.of_imbalance import compute_of_features_at

# Importa tu builder base existente
# Debe existir make_supervised_from_events(df, events, **kwargs) -> (X, y)
# con X conteniendo al menos 'idx' y 'zone_type'

from ppz.features.mvc import annotate_mvc_directional_flags, MvcFlagsParams

__all__ = ["make_supervised_from_events"]

def _session_tag(idx_in_session: pd.Series) -> pd.Series:
    tag = np.full(len(idx_in_session), "ASIA", dtype=object)
    tag[(idx_in_session >= 108) & (idx_in_session <= 185)] = "EU"
    tag[(idx_in_session >= 186) & (idx_in_session <= 264)] = "USA"
    return pd.Series(tag, index=idx_in_session.index)

def _ols_slope(y: np.ndarray) -> float:
    n = len(y)
    if n <= 1 or np.all(~np.isfinite(y)):
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = np.nanmean(y)
    num = np.nansum((x - x_mean) * (y - y_mean))
    den = np.nansum((x - x_mean) ** 2) + 1e-12
    return float(num / den)

def _count_prev_touches(df: pd.DataFrame, i: int, level: float, L: int, r_ticks: int, tick: float) -> int:
    j0 = max(0, i - L)
    dist = np.abs(df.loc[j0:i-1, "Close"].to_numpy(float) - level) / tick
    return int((dist <= r_ticks).sum())

def make_supervised_from_events(
    df: pd.DataFrame,
    events_labeled: pd.DataFrame,
    tick_size: float,
    n_short: int = 20,
    n_long: int = 60,
    L_prev_touches: int = 60,
    r_touch_ticks: int = 6,
    drop_none: bool = False,
    # === NUEVOS PARÁMETROS ===
    add_mvc: bool = True,
    add_orderflow: bool = True,
    of_k: float = 3.0,
    of_k_ext: float = 5.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye X,y por evento (una fila por 'idx' en events_labeled).
    Mantiene las ~features previas y (si se desea) añade:
      - MVC-based (pos/flags/body/wicks)
      - Order Flow: imbalances diagonales + stacked imbalances
    """
    ev = events_labeled.copy().reset_index(drop=True)

    # y (target)
    y = ev["label"].astype(str)
    if drop_none:
        m = y != "none"
        ev = ev.loc[m].reset_index(drop=True)
        y  = y.loc[m].reset_index(drop=True)

    # === BASE: 23 features existentes (asumimos que ya estaban aquí) ===
    # (este bloque es esquemático; deja tu código actual tal cual)
    X_list = []

    # ejemplo: zona one-hot ya existente
    ztypes = pd.get_dummies(ev["zone_type"], prefix="z").astype(np.int8)
    X_list.append(ztypes)

    # ejemplo: sesión one-hot si la tienes en ev o mapeable desde df
    # (si no, omítelo)
    if "session_tag" in ev.columns:
        sess = pd.get_dummies(ev["session_tag"], prefix="sess").astype(np.int8)
        X_list.append(sess)

    # distancias / slopes / ATR / etc. (tu bloque previo)
    # X_list.append( ... )

    # === NUEVO: MVC por vela del evento ===
    if add_mvc:
        mvc = compute_mvc_features(df).loc[ev["idx"]].reset_index(drop=True)
        X_list.append(mvc.add_prefix(""))  # ya viene con prefijo mvc_

    # === NUEVO: Order flow (solo en idx de eventos) ===
    if add_orderflow:
        of = compute_of_features_at(df, ev["idx"].to_numpy(), k=of_k, k_ext=of_k_ext)
        of = of.drop(columns=["idx"], errors="ignore")
        X_list.append(of)

    # concat final
    X = pd.concat(X_list, axis=1)

    # compactar tipos numéricos
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].astype(np.float32, errors="ignore")
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)

    # añade metadatos útiles (idx/zone_type) para downstream, pero no se usan como features
    X["idx"] = ev["idx"].astype(int).values
    X["zone_type"] = ev["zone_type"].astype(str).values

    return X, y


#--- new funciotns

def _one_hot_safe(series: pd.Series, prefix: str, categories: Iterable[str]) -> pd.DataFrame:
    """get_dummies con columnas garantizadas (faltantes -> 0)."""
    series = series.astype("category")
    d = pd.get_dummies(series, prefix=prefix, dtype=np.int8)
    need = [f"{prefix}_{c}" for c in categories]
    for col in need:
        if col not in d.columns:
            d[col] = 0
    return d[need]


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    return out


def make_supervised_from_events_mvcdir(
    df: pd.DataFrame,
    events_labeled: pd.DataFrame,
    *,
    mvc_params: Optional[MvcFlagsParams] = None,
    subset_sessions: Optional[Tuple[str, ...]] = None,  # p.ej. ("EU","USA")
    # kwargs que se pasan al builder base:
    tick_size: float = 0.25,
    n_short: int = 20,
    n_long: int = 60,
    L_prev_touches: int = 60,
    r_touch_ticks: int = 6,
    drop_none: bool = False,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye el dataset supervisado base y le añade:
      - exh_up, exh_down, reject_up, reject_down
      - side_{support,resistance}
      - vwap_{down,flat,up}
    Mantiene 'idx' y 'zone_type' para trazabilidad.
    """
    # 1) Opcional: filtra por sesión (si el label ya incluye 'session_tag')
    ev = events_labeled.copy()
    if subset_sessions is not None and "session_tag" in ev.columns:
        ev = ev[ev["session_tag"].isin(subset_sessions)].reset_index(drop=True)

    # 2) Anotar MVC direccional (aplica "primer toque" en estáticas)
    mvc_params = mvc_params or MvcFlagsParams(tick_size=tick_size)
    ev_mvc = annotate_mvc_directional_flags(df, ev, mvc_params)

    # 3) Dataset base (tus 23 features actuales, etc.)
    X_base, y = _base_builder(
        df, ev_mvc,
        tick_size=tick_size,
        n_short=n_short,
        n_long=n_long,
        L_prev_touches=L_prev_touches,
        r_touch_ticks=r_touch_ticks,
        drop_none=drop_none,
        **kwargs
    )

    # Asegura 'idx' y 'zone_type' (los devolvía tu builder original)
    if "idx" not in X_base.columns:
        X_base["idx"] = ev_mvc["idx"].astype(int).values
    if "zone_type" not in X_base.columns:
        X_base["zone_type"] = ev_mvc["zone_type"].astype(str).values

    # 4) Flags y one-hots
    flags_cols = ["idx","zone_type","exh_up","exh_down","reject_up","reject_down","side","vwap_slope_class"]
    F = ev_mvc[flags_cols].copy()

    # One-hot side / slope (columnas garantizadas)
    side_oh  = _one_hot_safe(F["side"].astype(str), "side", ["support","resistance"])
    slope_oh = _one_hot_safe(F["vwap_slope_class"].astype(str), "vwap", ["down","flat","up"])

    F_oh = pd.concat([F[["idx","zone_type","exh_up","exh_down","reject_up","reject_down"]].astype(np.int8),
                      side_oh, slope_oh], axis=1)

    # 5) Merge con features base
    X = X_base.merge(F_oh, on=["idx","zone_type"], how="left")

    # Relleno seguro para nuevas columnas
    add_cols = ["exh_up","exh_down","reject_up","reject_down",
                "side_support","side_resistance",
                "vwap_down","vwap_flat","vwap_up"]
    X = _ensure_cols(X, add_cols)

    # Tipos compactos
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].astype(np.float32, errors="ignore")
    for c in add_cols:
        X[c] = X[c].astype(np.int8)

    return X, pd.Series(y).astype(str)


def save_supervised_mvcdir(
    X: pd.DataFrame,
    y: pd.Series,
    out_path: str | Path,
) -> Path:
    """Guarda X+y juntos para reproducibilidad (Parquet zstd)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D = X.copy()
    D["target"] = y.values
    D.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return out_path