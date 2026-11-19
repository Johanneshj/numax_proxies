"""False-alarm thresholds for the ACF and CoV methods.

fap_tables.json holds, for bins of N = 4, 8, ..., 65536 independent chi^2_2 points, the
95th and 99th percentiles of the RAW (unsmoothed) statistic of white noise, from 1e6
simulated bins per N (scripts/make_fap_tables.py). They only depend on N. For the ACF
('all_lags') the statistic is MCC - mu0(N); mu0(N), the white-noise mean MCC, is stored too.
The threshold of a bin is interpolated in log(N) at its actual number of points.
Bins with N < 4 get no threshold (NaN: never significant); N above the table uses the
last entry (the thresholds have converged there).

These are PER-BIN thresholds: a spectrum with many bins will exceed one of them
somewhere by chance. They assume independent chi^2_2 points, i.e. a PSD that is not
oversampled and not averaged.
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
    """Read fap_tables.json (cached).

    Returns:
        {"N": array, "ACF": {"95": array, "99": array, "mu0": array},
        "CoV": {"95": array, "99": array}}.
    """
    raw = json.loads(resources.files("numax_proxies").joinpath("fap_tables.json").read_text())
    return {"N": np.asarray(raw["N"], dtype=float),
            **{m: {k: np.asarray(v, dtype=float) for k, v in raw[m].items()}
               for m in ("ACF", "CoV")}}


def table_threshold(method: str, n_points, threshold: str) -> NDArray:
    """Per-bin white-noise threshold from the FAP tables.

    Args:
        method: "ACF" or "CoV".
        n_points: number of points per bin (scalar or array).
        threshold: "95p0" or "99p0".

    Returns:
        Threshold per bin; NaN for N < 4. Interpolated in log N (CoV), or in log value vs
        log N (ACF: thresholds fall ~ N^-1/2, a straight line in log-log).
    """
    if threshold not in LEVELS:
        raise ValueError(f"FAP_threshold must be one of {list(LEVELS)}, got {threshold!r}")
    t = load_tables()
    return _interp_n(t["N"], t[method][f"{LEVELS[threshold]:g}"], n_points, log_y=method == "ACF")


def _interp_n(n_table, y_table, n_points, log_y: bool) -> NDArray:
    """Interpolate a table in log N (and log y if log_y); NaN below the table, last value above."""
    N = np.asarray(n_points, dtype=float)
    x = np.log(np.clip(N, 1, None))
    if log_y:
        out = np.exp(np.interp(x, np.log(n_table), np.log(y_table)))
    else:
        out = np.interp(x, np.log(n_table), y_table)
    return np.where(N >= n_table[0], out, np.nan)


def acf_noise_mean(n_points) -> NDArray:
    """White-noise mean of the MCC (mean |r(k)| over all lags) for bins of n_points points.

    Args:
        n_points: number of points per bin (scalar or array).

    Returns:
        mu0 per bin from the simulations, interpolated in log mu0 vs log N; NaN for N < 4.
    """
    t = load_tables()
    return _interp_n(t["N"], t["ACF"]["mu0"], n_points, log_y=True)


def add_faps(results: dict, method: str, cfg) -> dict:
    """Add the threshold curve "fap" to every grid setting (in place).

    Args:
        results: {setting: result dict}.
        method: "ACF" or "CoV".
        cfg: ACFConfig or COVConfig (FAP_threshold; collapse for the ACF).

    Returns:
        results. ACF delta_nu: z null_sd (analytic; the statistic already has its
        white-noise mean removed); otherwise the tables.
    """
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
    """Is a bin within n_fwhm envelope FWHM of numax significant?

    Args:
        centres: bin centres [muHz].
        raw: raw (unsmoothed) statistic per bin.
        fap: threshold per bin (or scalar).
        numax: frequency to check [muHz].
        n_fwhm: half-width of the window in envelope FWHM.

    Returns:
        True if any bin in the window has raw >= fap.
    """
    if numax is None or not np.isfinite(numax) or numax <= 0:
        return False
    x = np.asarray(centres, dtype=float)
    y = np.asarray(raw, dtype=float)
    f = np.broadcast_to(np.asarray(fap, dtype=float), y.shape)
    near = np.abs(x - numax) <= n_fwhm * envelope_fwhm(numax)
    return bool(np.any(near & np.isfinite(y) & np.isfinite(f) & (y >= f)))
