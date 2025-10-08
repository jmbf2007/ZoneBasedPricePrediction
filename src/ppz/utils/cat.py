# src/ppz/utils/cat.py
import numpy as np, pandas as pd

def one_hot_safe(series: pd.Series, prefix: str, categories):
    d = pd.get_dummies(series.astype("category"), prefix=prefix, dtype=np.int8)
    need = [f"{prefix}_{c}" for c in categories]
    for col in need:
        if col not in d.columns: d[col] = 0
    return d[need]
