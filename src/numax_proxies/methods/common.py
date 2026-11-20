"""Helpers shared by the ACF and CoV method classes."""
import gzip
import pickle
from pathlib import Path

import numpy as np
from uncertainties import ufloat


def fitted_numaxes(results: dict):
    """Fitted numax values of all grid settings.

    Args:
        results: {setting: result dict with "fit_numax" (ufloat)}.

    Returns:
        (numax array, error array).
    """
    objs = [d["fit_numax"] for d in results.values() if "fit_numax" in d]
    return np.array([x.n for x in objs], dtype=float), np.array([x.s for x in objs], dtype=float)


def median_estimate(results: dict):
    """Median numax and median error over the grid settings.

    Args:
        results: {setting: result dict with "fit_numax" (ufloat)}.

    Returns:
        ufloat(median numax, median error); NaN if no setting has a finite value.
    """
    numax, err = fitted_numaxes(results)
    ok = ~np.isnan(numax) & ~np.isnan(err)
    if not np.any(ok):
        return ufloat(np.nan, np.nan)
    return ufloat(np.median(numax[ok]), np.median(err[ok]))


def save_pickle(results: dict, path: Path) -> None:
    """Save results as a gzipped pickle.

    Args:
        results: results dictionary.
        path: output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_numax_estimates(results: dict, path: Path) -> None:
    """Save numax and its error of every grid setting as csv text.

    Args:
        results: {setting: result dict with "fit_numax"}.
        path: output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numax, err = fitted_numaxes(results)
    np.savetxt(path, np.column_stack([numax, err]), header="numax,numax_err",
               delimiter=",", fmt="%.4f")


def correct_fits(results: dict, frequency, median) -> None:
    """Background correction of every fitted numax (in place; see background.py).

    Args:
        results: {setting: result dict with "fit_numax" (ufloat)}.
        frequency: frequency grid [muHz].
        median: running median of the PSD on that grid.

    The uncorrected value is kept as "fit_numax_raw"; the formal error is unchanged.
    """
    from ..background import correct_numax
    for d in results.values():
        x = d.get("fit_numax")
        if x is None:
            continue
        d["fit_numax_raw"] = x
        if np.isfinite(x.n):
            d["fit_numax"] = ufloat(correct_numax(frequency, median, x.n), x.s)
