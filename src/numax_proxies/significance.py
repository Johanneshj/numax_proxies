"""Per-bin white-noise thresholds (FAP) for the ACF, ACV and CoV methods.

The thresholds come from simulated white-noise bins (fap_tables.json, made by
scripts/make_fap_tables.py) and depend only on the number of points N per bin. They are
per-bin thresholds and assume a PSD that is not oversampled or averaged.
"""
import json
from functools import lru_cache
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from .seismic import envelope_fwhm

# FAP_threshold name -> percentile
LEVELS = {"95p0": 95.0, "99p0": 99.0}


@lru_cache(maxsize=1)
def load_tables() -> dict:
    """Read fap_tables.json (cached)."""
    raw = json.loads(resources.files("numax_proxies").joinpath("fap_tables.json").read_text())
    return {"N": np.asarray(raw["N"], dtype=float),
            **{m: {k: np.asarray(v, dtype=float) for k, v in raw[m].items()}
               for m in ("ACF", "CoV", "ACV") if m in raw}}


def table_threshold(method: str, n_points, threshold: str) -> NDArray:
    """Threshold per bin for method "ACF", "ACV" or "CoV" at level "95p0" or "99p0",
    interpolated at each bin's number of points (NaN for N < 4).
    """
    if threshold not in LEVELS:
        raise ValueError(f"FAP_threshold must be one of {list(LEVELS)}, got {threshold!r}")
    t = load_tables()
    return _interp_n(t["N"], t[method][f"{LEVELS[threshold]:g}"], n_points, log_y=method != "CoV")


def _interp_n(n_table, y_table, n_points, log_y: bool) -> NDArray:
    """Interpolate a table in log N (and log y if log_y); NaN below the table, last value above."""
    N = np.asarray(n_points, dtype=float)
    x = np.log(np.clip(N, 1, None))
    if log_y:
        out = np.exp(np.interp(x, np.log(n_table), np.log(y_table)))
    else:
        out = np.interp(x, np.log(n_table), y_table)
    return np.where(N >= n_table[0], out, np.nan)


def acf_noise_mean(n_points, method: str = "ACF") -> NDArray:
    """White-noise mean mu0 of the ACF ('all_lags') or ACV statistic for bins of n_points points."""
    t = load_tables()
    return _interp_n(t["N"], t[method]["mu0"], n_points, log_y=True)


def add_faps(results: dict, method: str, cfg) -> dict:
    """Add the threshold curve "fap" to every grid setting (in place)."""
    from scipy.stats import norm

    gaussian = method == "ACF" and getattr(cfg, "collapse", "all_lags") == "delta_nu"
    for data in results.values():
        if gaussian:
            z = norm.ppf(LEVELS[cfg.FAP_threshold] / 100)
            data["fap"] = z * data["null_sd"]
        else:
            data["fap"] = table_threshold(method, data["N_data_points_in_bin"], cfg.FAP_threshold)
    return results


def significant_near(centres, raw, fap, numax, n_fwhm: float = 1.0) -> bool:
    """True if a bin within n_fwhm envelope FWHM of numax is above its threshold."""
    if numax is None or not np.isfinite(numax) or numax <= 0:
        return False
    x = np.asarray(centres, dtype=float)
    y = np.asarray(raw, dtype=float)
    f = np.broadcast_to(np.asarray(fap, dtype=float), y.shape)
    near = np.abs(x - numax) <= n_fwhm * envelope_fwhm(numax)
    return bool(np.any(near & np.isfinite(y) & np.isfinite(f) & (y >= f)))
