"""Monte Carlo harness: seed replication and parameter sweeps with bands."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model_v2 import ModelV2


def replicate(params, n_seeds=20, reducer=None):
    """Run a params config over n_seeds and return per-seed final-window metrics.

    reducer(history) -> dict of scalars. Default reports the standard set,
    averaged over the last 40 periods (or final value where a level).
    """
    if reducer is None:
        reducer = _default_reducer
    rows = []
    for s in range(n_seeds):
        h = ModelV2(replace(params, seed=s)).run()
        rows.append(reducer(h))
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


def _default_reducer(h):
    n = len(h.gini)
    w = slice(max(0, n - 40), n)
    return {
        "gini": float(np.mean(h.gini[w])),
        "top1": float(np.mean(h.top1_share[w])),
        "gov_nw": float(h.gov_nw[-1]),
        "row_nw": float(h.row_nw[-1]),
        "labour_share": float(np.mean(h.labour_share[w])),
        "Y": float(h.Y[-1]),
    }


def summarise(metrics):
    """mean and 95% band (2.5/97.5 pct) for each metric across seeds."""
    out = {}
    for k, v in metrics.items():
        out[k] = (float(np.mean(v)), float(np.percentile(v, 2.5)),
                  float(np.percentile(v, 97.5)))
    return out
