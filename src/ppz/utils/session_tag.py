import pandas as pd
import numpy as np


def add_session_tag_by_index(df, eu=(108,185), usa=(186,264), idx_col="idx_in_session"):
    out = df.copy()
    idx = out[idx_col].to_numpy()
    tag = np.full(len(out), "ASIA", dtype=object)
    tag[(idx>=eu[0]) & (idx<=eu[1])]  = "EU"
    tag[(idx>=usa[0]) & (idx<=usa[1])] = "USA"
    out["session_tag"] = tag
    return out