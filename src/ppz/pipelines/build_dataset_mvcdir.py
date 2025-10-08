# src/ppz/pipelines/build_dataset_mvcdir.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

# 👇 importa el builder BASE (el que ya tenías de 23 features)
from .build_dataset import make_supervised_from_events as _base_builder

# 👇 flags MVC direccionales y pendiente VWAP
from ppz.features.mvc import (
    annotate_mvc_directional_flags,
    MvcFlagsParams,
)

__all__ = [
    "make_supervised_from_events_mvcdir",
    "save_supervised_mvcdir",
]

def _one_hot_safe(series: pd.Series, prefix: str, categories: Iterable[str]) -> pd.DataFrame:
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
    subset_sessions: Optional[Tuple[str, ...]] = None,  # ej ("EU","USA")
    # kwargs al builder base:
    tick_size: float = 0.25,
    n_short: int = 20,
    n_long: int = 60,
    L_prev_touches: int = 60,
    r_touch_ticks: int = 6,
    drop_none: bool = False,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye el dataset base y añade:
      - exh_up, exh_down, reject_up, reject_down
      - side_{support,resistance}
      - vwap_{down,flat,up}
    Mantiene 'idx' y 'zone_type'.
    """
    ev = events_labeled.copy()
    if subset_sessions is not None and "session_tag" in ev.columns:
        ev = ev[ev["session_tag"].isin(subset_sessions)].reset_index(drop=True)

    mvc_params = mvc_params or MvcFlagsParams(tick_size=tick_size)
    ev_mvc = annotate_mvc_directional_flags(df, ev, mvc_params)

    # builder base (23 feats)
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

    if "idx" not in X_base.columns:
        X_base["idx"] = ev_mvc["idx"].astype(int).values
    if "zone_type" not in X_base.columns:
        X_base["zone_type"] = ev_mvc["zone_type"].astype(str).values

    flags_cols = ["idx","zone_type","exh_up","exh_down","reject_up","reject_down","side","vwap_slope_class"]
    F = ev_mvc[flags_cols].copy()

    side_oh  = _one_hot_safe(F["side"].astype(str), "side", ["support","resistance"])
    slope_oh = _one_hot_safe(F["vwap_slope_class"].astype(str), "vwap", ["down","flat","up"])

    F_oh = pd.concat(
        [
            F[["idx","zone_type","exh_up","exh_down","reject_up","reject_down"]].astype(np.int8),
            side_oh,
            slope_oh,
        ],
        axis=1
    )

    X = X_base.merge(F_oh, on=["idx","zone_type"], how="left")

    add_cols = [
        "exh_up","exh_down","reject_up","reject_down",
        "side_support","side_resistance",
        "vwap_down","vwap_flat","vwap_up",
    ]
    X = _ensure_cols(X, add_cols)

    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].astype(np.float32, errors="ignore")
    for c in add_cols:
        X[c] = X[c].astype(np.int8)

    return X, pd.Series(y).astype(str)

def save_supervised_mvcdir(X: pd.DataFrame, y: pd.Series, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D = X.copy()
    D["target"] = y.values
    D.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return out_path
