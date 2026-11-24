"""Comb scan: checks that a fitted numax sits on a regular comb of modes, and re-centres it.

Around the (power-excess-shifted) numax, the relative power is cut into windows of constant
width. Each window's ACF is collapsed onto lag bands at multiples of the spacing D (the
"integer" profile at D and 2D decides; the half, combined, anti and total profiles are
diagnostics), converted to z against white noise (analytic, corrected with
comb_scan_null.json) and fitted with a Gaussian. recentre() repeats the scan on the fitted
centre until it converges, or rejects the numax if the peak z is below the threshold.

The ACV check uses the same windows and bands, but on the autocovariance (the ACF not divided
by the window's own variance): it also responds to power, so it survives a few dominant modes,
but also to tall spikes.
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
NOISE_VAR = 1.0 / np.log(2.0) ** 2   # variance of P / median for chi^2_2 white noise
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
    """Lag bands (lo, hi) and weights of a profile for the trial spacings d_pts [points]."""
    mult = [m for m, _ in PROFILES[name]]
    lo, hi = band_edges(d_pts, mult, hw)
    return lo, hi, np.array([w for _, w in PROFILES[name]])


SELECT = 1.0           # D* from the windows within +- SELECT x FWHM of numax
N_LAG_MAP = 200        # lag points of the 2D ACF map


def relative_power(frequency, power):
    """P / running median - 1."""
    return power / running_median(frequency, power) - 1.0


def _autocov(x):
    """ACF of x after removing a straight line: (r(k) = ACF(k) / ACF(0), variance per point)."""
    t = np.arange(x.size) - 0.5 * (x.size - 1)          # centred index: slope and mean decouple
    x = x - x.mean() - t * (np.dot(t, x) / np.dot(t, t))
    nfft = 1 << int(np.ceil(np.log2(2 * x.size)))
    c = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:x.size]
    return c / c[0], c[0] / x.size


def _autocorr(x):
    """r(k) = ACF(k) / ACF(0) of x after removing a straight line."""
    return _autocov(x)[0]


def band_edges(d_pts, multiples, hw=HALF):
    """Lag bands m D +- hw D in points, (lo, hi) of shape (n_bands, n_D)."""
    lo = np.array([np.ceil((m - hw) * d_pts) for m in multiples]).astype(int)
    hi = np.array([np.floor((m + hw) * d_pts) for m in multiples]).astype(int)
    return lo, np.maximum(hi, lo)


def excess_peak(frequency, rel_power, numax, span):
    """Frequency of the maximum of the smoothed P / median within numax +- span FWHM (NaN if too few points)."""
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
    """Weighted mean of the band means of r for every trial D."""
    w = np.ones(len(lo)) if w is None else w
    cs = np.concatenate([[0.0], np.cumsum(r)])
    return sum(w[b] * (cs[hi[b] + 1] - cs[lo[b]]) / (hi[b] - lo[b] + 1)
               for b in range(len(lo))) / len(lo)


@lru_cache(maxsize=1)
def _null_tables():
    """Interpolators of the line-removal correction (comb_scan_null.json), per profile."""
    t = json.loads(resources.files("numax_proxies").joinpath("comb_scan_null.json").read_text())
    axes = (np.log10(t["n"]), np.asarray(t["x"]), np.asarray(t["half_width"]))
    return {name: (RegularGridInterpolator(axes, np.asarray(v["bias"])),
                   RegularGridInterpolator(axes, np.asarray(v["sd_ratio"])), axes)
            for name, v in t["profiles"].items()}


def null_mean_sd(n, d_pts, lo, hi, profile=DECIDES, w=None, hw=HALF):
    """White-noise mean and sd of a profile for windows of n points, per trial D."""
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
    """Gaussian fit (amplitude >= 0) to a z profile; (popt, pcov), or (None, None) if it fails."""
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
    """One comb scan around numax.

    Returns a dict with the windows (centres), the deciding profile (z, popt, numax_comb, peak_z),
    all profiles, the ACV check (acv), the shifted numax (scan_centre) and, with with_map, the
    local 2D ACF for plotting; None if the scan does not fit in the spectrum.
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

    RV = [_autocov(rel_power[a:a + n]) for a in starts]
    R = [r for r, _ in RV]
    var = np.array([v for _, v in RV]) / (NOISE_VAR * (n - 2) / n)   # 1 for white noise
    k = np.arange(n)
    norm = n / np.sqrt(np.maximum(n - k, 1))                 # r -> r / sd(r) for white noise

    # D*: the band at lag D alone, averaged over the windows near the scan centre
    lo, hi = band_edges(d_pts, [1.0])
    mu0, sd0 = null_mean_sd(n, d_pts, lo, hi, DECIDES, np.ones(1))
    za = np.array([(_collapse(r, lo, hi) - mu0) / sd0 for r in R])
    j = int(np.argmax(za[near].mean(axis=0)))

    def fitted(z):
        p, cov = _fit(centres, z, numax, fwhm)
        return dict(z=z, popt=p, cov=cov, peak_z=np.nan if p is None else float(p[0] + p[3]),
                    centre=ufloat(p[2], np.sqrt(cov[2, 2])) if p is not None
                    else ufloat(np.nan, np.nan))

    profiles = {}
    for name, (lo, hi, w) in bands.items():
        mu0, sd0 = null_mean_sd(n, d_pts, lo, hi, name, w)
        c = np.array([_collapse(r, lo, hi, w)[j] for r in R])
        profiles[name] = fitted((c - mu0[j]) / sd0[j])
        if name == DECIDES:                                  # the ACV check: same bands, not
            acv = fitted((c * var - mu0[j]) / sd0[j])        # divided by the window variance
    lag = np.linspace(0.0, width, N_LAG_MAP)                 # [Dnu_s]
    amap = (np.array([np.interp(lag * dnu / df, k, r * norm) for r in R]).T if with_map else None)
    main = profiles[DECIDES]
    centre = main["centre"]
    return dict(centres=centres, z=main["z"], lag=lag, map=amap, dnu=dnu, D=d_pts[j] * df,
                fwhm=fwhm, numax=guess, scan_centre=float(numax), width=float(width),
                popt=main["popt"], numax_comb=centre, peak_z=main["peak_z"],
                offset=centre.n / guess - 1.0, profiles=profiles, var=var, acv=acv)


def significant(r, threshold) -> bool:
    """True if the integer profile has a positive Gaussian with peak z >= threshold."""
    return (r is not None and r["popt"] is not None and r["popt"][0] > 0
            and r["peak_z"] >= threshold)


def acv_significant(r, threshold) -> bool:
    """True if the ACV check has a positive Gaussian with peak z >= threshold."""
    return (r is not None and r["acv"]["popt"] is not None and r["acv"]["popt"][0] > 0
            and r["acv"]["peak_z"] >= threshold)


def recentre(frequency, rel_power, numax, width=4.0, span=2.0, threshold=5.0, tolerance=0.1,
             max_iterations=5, with_map=True, aim_at="excess"):
    """Repeat the comb scan on its fitted centre until it moves by <= tolerance x FWHM.

    Returns dict(status = "converged" | "not_converged" | "rejected", numax, start, scans).
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
