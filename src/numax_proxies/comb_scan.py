"""Comb scan: a sanity check of a fitted numax from a local 2D ACF with constant windows.

Around the tested numax (Dnu_s = Delta-nu(numax) from the scaling relation, FWHM = envelope
FWHM):
  1. windows of constant width `width` x Dnu_s, centres every Dnu_s / 4 over numax +- `span` FWHM
  2. per window: relative power minus a straight line, linear ACF r(k) = ACF(k) / ACF(0)
  3. collapse onto lag bands m D +- 0.1 D, as the weighted mean of the band means
     (weight +1: correlation expected, -1: anticorrelation expected, sign flipped), profiles:
       integer      m = 1, 2                          modes repeating every Dnu (l = 0, 2; p-like l = 1)
       half         m = 1/2, 3/2                      l = 1 halfway between the radial modes
       combined     m = 1/2, 1, 3/2, 2
       anti         m = 1/4, 3/4, 5/4, 7/4, 9/4  (-1)  the dips between the peaks, sign flipped
       total        combined (+1) and anti (-1)
     z = (c - null mean) / null sd. White-noise null: analytic for a mean-removed window,
     corrected for the line removal with comb_scan_null.json (scripts/make_comb_scan_null.py)
  4. Aim: the scan is first re-centred on the power excess (maximum of P / median smoothed
     over one envelope FWHM within numax +- span FWHM): windows, Dnu_s and lag units follow
     that frequency. D* = the trial D in [0.7, 1.3] Dnu_s with the largest z of the band at
     lag D alone, averaged over the windows within +- 1 FWHM of the scan centre.
  5. Gaussian (with offset) fit to each profile at D*. The INTEGER profile decides (accept,
     re-centre, numax): it survives mixed dipole modes (subgiants), whose l = 1 modes are off
     the half-way positions. The other profiles are diagnostics. Note: the dips sit at the
     quarter positions only when l = 1 lies halfway; a broad negative feature (dip, artefact)
     is anticorrelated at all moderate lags and makes "anti" positive too.

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
_PEAKS = (0.5, 1.0, 1.5, 2.0)
_DIPS = (0.25, 0.75, 1.25, 1.75, 2.25)
PROFILES = {   # name: ((multiple of D, weight), ...)
    "integer": ((1.0, 1.0), (2.0, 1.0)),
    "half": ((0.5, 1.0), (1.5, 1.0)),
    "combined": tuple((m, 1.0) for m in _PEAKS),
    "anti": tuple((m, -1.0) for m in _DIPS),
    "total": tuple((m, 1.0) for m in _PEAKS) + tuple((m, -1.0) for m in _DIPS),
}
DECIDES = "integer"
MIN_WIDTH = (max(m for bands in PROFILES.values() for m, _ in bands) + HALF) * D_RANGE[1]   # [Dnu_s]


def profile_bands(d_pts, name, hw=HALF):
    """(lo, hi, weights) of a profile for trial spacings d_pts [points], band half-width hw [D]."""
    mult = [m for m, _ in PROFILES[name]]
    lo, hi = band_edges(d_pts, mult, hw)
    return lo, hi, np.array([w for _, w in PROFILES[name]])


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


def band_edges(d_pts, multiples, hw=HALF):
    """Lag bands m D +- hw D in points: (lo, hi), each (n_bands, n_D); at least one lag each."""
    lo = np.array([np.ceil((m - hw) * d_pts) for m in multiples]).astype(int)
    hi = np.array([np.floor((m + hw) * d_pts) for m in multiples]).astype(int)
    return lo, np.maximum(hi, lo)


def excess_peak(frequency, rel_power, numax, span):
    """Frequency of the maximum of P / median smoothed over one envelope FWHM (running mean),
    within numax +- span x FWHM: where the power excess is.

    Returns:
        Frequency [muHz], or NaN if the range holds too few points.
    """
    fwhm = float(envelope_fwhm(numax))
    df = frequency[1] - frequency[0]
    k = max(1, int(round(fwhm / df)))
    lo, hi = numax - span * fwhm, numax + span * fwhm
    sel = np.flatnonzero((frequency >= lo - fwhm) & (frequency <= hi + fwhm))
    if sel.size < 3 * k:
        return np.nan
    cs = np.concatenate([[0.0], np.cumsum(rel_power[sel])])
    sm = (cs[k:] - cs[:-k]) / k                              # running mean, length sel.size - k + 1
    f_mid = frequency[sel[: sm.size]] + 0.5 * (k - 1) * df
    ok = (f_mid >= max(lo, frequency[0])) & (f_mid <= hi)
    if not ok.any():
        return np.nan
    return float(f_mid[ok][np.argmax(sm[ok])])


def _collapse(r, lo, hi, w=None):
    """c(D) for every trial D: weighted mean of the band means (lo, hi: (n_bands, n_D) lag
    indices; w: weight per band, default +1)."""
    w = np.ones(len(lo)) if w is None else w
    cs = np.concatenate([[0.0], np.cumsum(r)])
    return sum(w[b] * (cs[hi[b] + 1] - cs[lo[b]]) / (hi[b] - lo[b] + 1)
               for b in range(len(lo))) / len(lo)


@lru_cache(maxsize=1)
def _null_tables():
    """{profile: (bias, sd ratio interpolators over (log10 n, D / n, half-width), axes)}."""
    t = json.loads(resources.files("numax_proxies").joinpath("comb_scan_null.json").read_text())
    axes = (np.log10(t["n"]), np.asarray(t["x"]), np.asarray(t["half_width"]))
    return {name: (RegularGridInterpolator(axes, np.asarray(v["bias"])),
                   RegularGridInterpolator(axes, np.asarray(v["sd_ratio"])), axes)
            for name, v in t["profiles"].items()}


def null_mean_sd(n, d_pts, lo, hi, profile=DECIDES, w=None, hw=HALF):
    """White-noise mean and sd of c(D) for a line-removed window of n points.

    Analytic for a mean-removed window (E r(k) = -(n-k)/n^2, Var r(k) = (n-k)/n^2, lags
    conditioned on the sum rule), corrected with the simulated table comb_scan_null.json.

    Args:
        n: window length [points].
        d_pts: trial spacings D [points].
        lo, hi: (n_bands, n_D) lag bands (band_edges).
        profile: name in PROFILES (selects the line-removal correction).
        w: weight per band (default: the profile's weights).
        hw: band half-width [D] (selects the line-removal correction).

    Returns:
        (mean, sd) per trial D.
    """
    k = np.arange(n)
    cs = np.concatenate([[0.0], np.cumsum((n - k) / n ** 2)])
    nb = len(lo)
    w = np.array([wt for _, wt in PROFILES[profile]]) if w is None else w
    mean = np.zeros(d_pts.size)
    var0 = np.zeros(d_pts.size)
    for b in range(nb):                       # lags of different bands: independent, plus the
        cnt = hi[b] - lo[b] + 1               # sum-rule term below (linear in the weights)
        s = cs[hi[b] + 1] - cs[lo[b]]
        mean -= w[b] * s / cnt / nb
        var0 += w[b] ** 2 * s / cnt ** 2 / nb ** 2
    sd = np.sqrt(np.maximum(var0 - mean ** 2 * 2 * n / (n - 1), 1e-30))
    bias, ratio, (ln, x, hws) = _null_tables()[profile]
    pts = np.column_stack([np.full(d_pts.size, np.clip(np.log10(n), ln[0], ln[-1])),
                           np.clip(d_pts / n, x[0], x[-1]),
                           np.full(d_pts.size, np.clip(hw, hws[0], hws[-1]))])
    sd_line = ratio(pts) * sd
    return mean + bias(pts) * sd_line, sd_line


def _fit(centres, z, numax, fwhm):
    """Gaussian (with offset) fit to the profile, amplitude >= 0 (a comb envelope is a peak).

    Started at the profile maximum (running mean over 5 windows) and at the tested numax; the
    fit with the smaller squared residual is kept.

    Returns:
        (popt, pcov), or (None, None) if no fit succeeds or the centre is at the edge.
    """
    lo, hi = centres.min(), centres.max()
    sig = fwhm / FWHM_TO_SIGMA
    smooth = np.convolve(z, np.ones(5) / 5, mode="same")
    starts = {float(centres[np.argmax(smooth)]), float(np.clip(numax, lo, hi))}
    base = float(np.median(z))
    best = (np.inf, None, None)
    for c0 in starts:
        a0 = max(float(np.interp(c0, centres, smooth)) - base, 1e-3)
        try:
            p, cov = curve_fit(gaussian, centres, z, jac=gaussian_jac, p0=[a0, sig, c0, base],
                               bounds=([0.0, 0.25 * sig, lo, -np.inf], [np.inf, 4 * sig, hi, np.inf]))
        except (RuntimeError, ValueError):
            continue
        if np.isclose(p[2], lo, rtol=1e-6) or np.isclose(p[2], hi, rtol=1e-6):
            continue
        rss = float(np.sum((z - gaussian(centres, *p)) ** 2))
        if rss < best[0]:
            best = (rss, p, cov)
    return best[1], best[2]


def comb_scan(frequency, rel_power, numax, width=4.0, span=2.0, with_map=True, aim_at="excess"):
    """Comb scan around numax.

    Args:
        frequency: uniform frequency grid [muHz].
        rel_power: relative power P / running median - 1 (see relative_power).
        numax: guess numax [muHz] (the ACF / CoV estimate).
        width: window width [Delta-nu]; at least MIN_WIDTH (the 9/4 D band must fit).
        span: scan range +- span x envelope FWHM.
        with_map: also return the local 2D ACF (r / sd(r) per lag and window) for plotting.
        aim_at: "excess" (default): the scan is re-centred on the power-excess peak within
            numax +- span FWHM (the shifted numax); "numax": centred on the guess (test option).

    Returns:
        dict (centres, z, lag, map, dnu, D, fwhm, numax (the guess), scan_centre (the shifted
        numax), width, popt, numax_comb (ufloat; NaN if no fit), peak_z, offset = numax_comb /
        numax - 1) for the deciding (integer) profile, and profiles = {name: dict(z, popt, cov,
        peak_z, centre)} for every profile; None if no window fits.
    """
    frequency = np.asarray(frequency, dtype=float)
    rel_power = np.asarray(rel_power, dtype=float)
    if not np.isfinite(numax) or numax <= 0:
        return None
    guess = float(numax)
    df = frequency[1] - frequency[0]
    if aim_at == "excess":                                   # re-centre on the power excess
        numax = excess_peak(frequency, rel_power, guess, span)
        if not np.isfinite(numax):
            return None
    dnu, fwhm = float(delta_nu(numax)), float(envelope_fwhm(numax))
    n = int(round(width * dnu / df))
    d_pts = np.arange(int(D_RANGE[0] * dnu / df), int(D_RANGE[1] * dnu / df) + 1)
    bands = {name: profile_bands(d_pts, name) for name in PROFILES}
    if n < 10 or any(hi.max() >= n or lo.min() < 1 for lo, hi, _ in bands.values()):
        return None

    centres = np.arange(numax - span * fwhm, numax + span * fwhm + 1e-9, STEP * dnu)
    starts = np.searchsorted(frequency, centres - 0.5 * width * dnu)
    ok = (centres - 0.5 * width * dnu >= frequency[0]) & (starts + n <= frequency.size)
    centres, starts = centres[ok], starts[ok]
    if centres.size < 5:
        return None
    near = np.abs(centres - numax) <= SELECT * fwhm
    if not near.any():
        near = np.ones(centres.size, dtype=bool)

    R = [_autocorr(rel_power[a:a + n]) for a in starts]
    k = np.arange(n)
    norm = n / np.sqrt(np.maximum(n - k, 1))                 # r -> r / sd(r) for white noise

    # D*: the band at lag D alone, averaged over the windows near the scan centre
    lo, hi = band_edges(d_pts, [1.0])
    mu0, sd0 = null_mean_sd(n, d_pts, lo, hi, DECIDES, np.ones(1))
    za = np.array([(_collapse(r, lo, hi) - mu0) / sd0 for r in R])
    j = int(np.argmax(za[near].mean(axis=0)))

    profiles = {}
    for name, (lo, hi, w) in bands.items():
        mu0, sd0 = null_mean_sd(n, d_pts, lo, hi, name, w)
        z = np.array([(_collapse(r, lo, hi, w)[j] - mu0[j]) / sd0[j] for r in R])
        p, cov = _fit(centres, z, numax, fwhm)
        profiles[name] = dict(z=z, popt=p, cov=cov,
                              peak_z=np.nan if p is None else float(p[0] + p[3]),
                              centre=ufloat(p[2], np.sqrt(cov[2, 2])) if p is not None
                              else ufloat(np.nan, np.nan))
    lag = np.linspace(0.0, width, N_LAG_MAP)                 # [Dnu_s]
    amap = (np.array([np.interp(lag * dnu / df, k, r * norm) for r in R]).T if with_map else None)
    main = profiles[DECIDES]
    centre = main["centre"]
    return dict(centres=centres, z=main["z"], lag=lag, map=amap, dnu=dnu, D=d_pts[j] * df,
                fwhm=fwhm, numax=guess, scan_centre=float(numax), width=float(width),
                popt=main["popt"], numax_comb=centre, peak_z=main["peak_z"],
                offset=centre.n / guess - 1.0, profiles=profiles)


def significant(r, threshold) -> bool:
    """True if the deciding (integer) profile has a positive Gaussian with peak z >= threshold."""
    return (r is not None and r["popt"] is not None and r["popt"][0] > 0
            and r["peak_z"] >= threshold)


def recentre(frequency, rel_power, numax, width=4.0, span=2.0, threshold=5.0, tolerance=0.1,
             max_iterations=5, with_map=True, aim_at="excess"):
    """Comb scan around numax, re-centred on the fitted centre until it converges.

    Args:
        frequency, rel_power, width, span, with_map, aim_at: see comb_scan.
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
        r = comb_scan(frequency, rel_power, nu, width, span, with_map, aim_at)
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
