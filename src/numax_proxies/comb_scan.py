"""Comb scan: a sanity check of a fitted numax from a local 2D ACF with constant windows.

Around the tested numax (Dnu_s = Delta-nu(numax) from the scaling relation, FWHM = envelope
FWHM):
  1. windows of constant width `width` x Dnu_s, centres every Dnu_s / 4 over numax +- `span` FWHM
  2. per window: relative power minus a straight line, linear ACF r(k) = ACF(k) / ACF(0)
  3. collapse: c(D) = mean of the band means of r over D/2 +- 0.1 D and D +- 0.1 D, as
     z = (c - null mean) / null sd. White-noise null: analytic for a mean-removed window,
     corrected for the line removal with comb_scan_null.json (scripts/make_comb_scan_null.py)
  4. D* = the trial D in [0.7, 1.3] Dnu_s with the largest z averaged over the windows within
     +- 1 FWHM of numax (the comb spacing is located from the data)
  5. Gaussian (with offset) fit to the profile z(D*) over all windows; its centre is where
     the comb is strongest.

A real comb gives a positive Gaussian; the fitted centre lands on the envelope even when the
tested numax is off (as long as the envelope is inside the scan). Artefacts and humps give
no positive peak (a broad excess or dip anticorrelates at these lags).

recentre(): repeat the scan around the fitted centre (new Dnu_s from the scaling relation)
until the centre moves by less than `tolerance` x FWHM; the envelope is significant if the
fitted Gaussian is positive with peak z >= `threshold`.
"""
import json
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from uncertainties import ufloat

from .binning import running_median
from .fitting import FWHM_TO_SIGMA, gaussian, gaussian_jac
from .seismic import delta_nu, envelope_fwhm

STEP = 0.25            # window step [Dnu_s]
HALF = 0.1             # band half-width [D]
D_RANGE = (0.7, 1.3)   # trial spacings [Dnu_s]
SELECT = 1.0           # D* from the windows within +- SELECT x FWHM of numax
N_LAG_MAP = 200        # lag points of the 2D ACF map


def relative_power(frequency, power):
    """P / running median - 1 (default running-median window)."""
    return power / running_median(frequency, power) - 1.0


def _autocorr(x):
    """Linear (zero-padded) autocorrelation r(k) = ACF(k) / ACF(0) of x minus a straight line."""
    t = np.arange(x.size) - 0.5 * (x.size - 1)          # centred index: slope and mean decouple
    x = x - x.mean() - t * (np.dot(t, x) / np.dot(t, t))
    nfft = 1 << int(np.ceil(np.log2(2 * x.size)))
    c = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:x.size]
    return c / c[0]


def _collapse(r, lo, hi):
    """c(D) for every trial D: mean of the two band means (lo, hi: (2, n_D) lag indices)."""
    cs = np.concatenate([[0.0], np.cumsum(r)])
    return 0.5 * sum((cs[hi[b] + 1] - cs[lo[b]]) / (hi[b] - lo[b] + 1) for b in range(2))


@lru_cache(maxsize=1)
def _null_table():
    """Interpolators (log10 n, D / n) -> bias, sd ratio of the line-removed null."""
    t = json.loads(resources.files("numax_proxies").joinpath("comb_scan_null.json").read_text())
    axes = (np.log10(t["n"]), np.asarray(t["x"]))
    return (RegularGridInterpolator(axes, np.asarray(t["bias"])),
            RegularGridInterpolator(axes, np.asarray(t["sd_ratio"])), axes)


def null_mean_sd(n, d_pts, lo, hi):
    """White-noise mean and sd of c(D) for a line-removed window of n points.

    Analytic for a mean-removed window (E r(k) = -(n-k)/n^2, Var r(k) = (n-k)/n^2, lags
    conditioned on the sum rule), corrected with the simulated table comb_scan_null.json.

    Args:
        n: window length [points].
        d_pts: trial spacings D [points].
        lo, hi: (2, n_D) lag bands (see comb_scan).

    Returns:
        (mean, sd) per trial D.
    """
    k = np.arange(n)
    cs = np.concatenate([[0.0], np.cumsum((n - k) / n ** 2)])
    mean = np.zeros(d_pts.size)
    var0 = np.zeros(d_pts.size)
    for b in range(2):
        cnt = hi[b] - lo[b] + 1
        s = cs[hi[b] + 1] - cs[lo[b]]
        mean -= 0.5 * s / cnt
        var0 += 0.25 * s / cnt ** 2
    sd = np.sqrt(np.maximum(var0 - mean ** 2 * 2 * n / (n - 1), 1e-30))
    bias, ratio, (ln, x) = _null_table()
    pts = np.column_stack([np.full(d_pts.size, np.clip(np.log10(n), ln[0], ln[-1])),
                           np.clip(d_pts / n, x[0], x[-1])])
    sd_line = ratio(pts) * sd
    return mean + bias(pts) * sd_line, sd_line


def _fit(centres, z, numax, fwhm):
    """Bounded Gaussian with offset; None if the fit fails or its centre is at the edge."""
    lo, hi = centres.min(), centres.max()
    sig = fwhm / FWHM_TO_SIGMA
    try:
        p, cov = curve_fit(gaussian, centres, z, jac=gaussian_jac, p0=[max(z.max(), 1.0), sig, numax, 0.0],
                           bounds=([-np.inf, 0.25 * sig, lo, -np.inf], [np.inf, 4 * sig, hi, np.inf]))
    except (RuntimeError, ValueError):
        return None, None
    if np.isclose(p[2], lo, rtol=1e-6) or np.isclose(p[2], hi, rtol=1e-6):
        return None, None
    return p, cov


def comb_scan(frequency, rel_power, numax, width=3.0, span=2.0, with_map=True):
    """Comb scan around numax.

    Args:
        frequency: uniform frequency grid [muHz].
        rel_power: relative power P / running median - 1 (see relative_power).
        numax: tested numax [muHz].
        width: window width [Delta-nu(numax)].
        span: scan range numax +- span x envelope FWHM.
        with_map: also return the local 2D ACF (r / sd(r) per lag and window) for plotting.

    Returns:
        dict (centres, z, lag, map, dnu, D, fwhm, numax, popt, numax_comb (ufloat; NaN if no
        fit), peak_z, offset = numax_comb / numax - 1), or None if no window fits.
    """
    frequency = np.asarray(frequency, dtype=float)
    rel_power = np.asarray(rel_power, dtype=float)
    if not np.isfinite(numax) or numax <= 0:
        return None
    dnu, fwhm = float(delta_nu(numax)), float(envelope_fwhm(numax))
    df = frequency[1] - frequency[0]
    n = int(round(width * dnu / df))
    d_pts = np.arange(int(D_RANGE[0] * dnu / df), int(D_RANGE[1] * dnu / df) + 1)
    lo = np.array([np.ceil((m - HALF) * d_pts) for m in (0.5, 1.0)]).astype(int)
    hi = np.array([np.floor((m + HALF) * d_pts) for m in (0.5, 1.0)]).astype(int)
    if n < 10 or hi.max() >= n or lo.min() < 1:
        return None

    centres = np.arange(numax - span * fwhm, numax + span * fwhm + 1e-9, STEP * dnu)
    starts = np.searchsorted(frequency, centres - 0.5 * width * dnu)
    ok = (centres - 0.5 * width * dnu >= frequency[0]) & (starts + n <= frequency.size)
    centres, starts = centres[ok], starts[ok]
    if centres.size < 5:
        return None

    mu0, sd0 = null_mean_sd(n, d_pts, lo, hi)

    lag = np.linspace(0.0, width, N_LAG_MAP)     # [Dnu_s]
    k = np.arange(n)
    Z, amap = [], []
    for a in starts:
        r = _autocorr(rel_power[a:a + n])
        Z.append((_collapse(r, lo, hi) - mu0) / sd0)
        if with_map:
            amap.append(np.interp(lag * dnu / df, k, r * n / np.sqrt(np.maximum(n - k, 1))))
    Z = np.array(Z)
    near = np.abs(centres - numax) <= SELECT * fwhm
    j = int(np.argmax(Z[near].mean(axis=0))) if near.any() else int(np.argmax(Z.mean(axis=0)))
    z = Z[:, j]

    p, cov = _fit(centres, z, numax, fwhm)
    centre = ufloat(p[2], np.sqrt(cov[2, 2])) if p is not None else ufloat(np.nan, np.nan)
    return dict(centres=centres, z=z, lag=lag, map=np.array(amap).T if with_map else None, dnu=dnu,
                D=d_pts[j] * df, fwhm=fwhm, numax=float(numax), width=float(width), popt=p,
                numax_comb=centre, peak_z=np.nan if p is None else float(p[0] + p[3]),
                offset=centre.n / numax - 1.0)


def significant(r, threshold) -> bool:
    """True if the scan has a positive fitted Gaussian with peak z >= threshold."""
    return (r is not None and r["popt"] is not None and r["popt"][0] > 0
            and r["peak_z"] >= threshold)


def recentre(frequency, rel_power, numax, width=3.0, span=2.0, threshold=5.0, tolerance=0.1,
             max_iterations=5, with_map=True):
    """Comb scan around numax, re-centred on the fitted centre until it converges.

    Args:
        frequency, rel_power, width, span, with_map: see comb_scan.
        numax: starting numax [muHz].
        threshold: minimum peak z of a significant envelope.
        tolerance: converged when the centre moves by <= tolerance x envelope FWHM.
        max_iterations: scans at most.

    Returns:
        dict: status ("converged", "not_converged" (last centre used) or "rejected"),
        numax (ufloat; NaN if rejected), start (numax), scans (list of comb_scan dicts).
    """
    scans, nu = [], float(numax)
    for _ in range(int(max_iterations)):
        r = comb_scan(frequency, rel_power, nu, width, span, with_map)
        if r is not None:
            scans.append(r)
        if not significant(r, threshold):
            return dict(status="rejected", numax=ufloat(np.nan, np.nan), start=float(numax),
                        scans=scans)
        centre = r["numax_comb"]
        if abs(centre.n - nu) <= tolerance * r["fwhm"]:
            return dict(status="converged", numax=centre, start=float(numax), scans=scans)
        nu = centre.n
    return dict(status="not_converged", numax=scans[-1]["numax_comb"], start=float(numax),
                scans=scans)
