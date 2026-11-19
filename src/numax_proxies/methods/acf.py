"""numax from the collapsed 2D autocorrelation function (Huber+2009, Viani+2019).

Steps, for every (overlap_scale, width_factor) on the grid:
  1. relative power  (P - median) / median, with a running median over
     median_filter_factor x envelope FWHM
  2. bins of width_factor x Delta-nu(nu), overlap_scale bins per bin width
  3. per bin, autocorrelation r(k) = ACF(k) / ACF(0) of the centred relative power
     (regularity within the bin, not its variance), collapsed to one number
     (ACF_CONFIG.collapse) minus its white-noise mean:
       all_lags : mean |r(k)| over all lags 1..n-1 (MCC) - mu0(n)   (mu0 from simulations)
       delta_nu : mean r(k) over the lags within +-dnu_window of m x Delta-nu(bin centre),
                  m in dnu_multiples, - its analytic white-noise mean
  4. smoothing over smoothing_factor x envelope FWHM (one curve per smoothing factor)
  5. significance: raw CACF compared with a per-bin FAP threshold
  6. Gaussian fit to the smoothed CACF -> numax
"""
import gzip
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.fft
import rocket_fft  # noqa: F401  (registers scipy.fft for Numba)
from numba import njit, prange
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from uncertainties import ufloat

from ..binning import (as_float_list, log_numax_bins, replace_below_min_freq, running_median,
                       smooth_over_envelope)
from ..fitting import FWHM_TO_SIGMA, gaussian
from ..seismic import delta_nu, envelope_fwhm
from ..significance import acf_noise_mean, add_faps, significant_near
from ..threads import numba_threads
from .common import median_estimate, save_numax_estimates, save_pickle

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Collapsed ACF
# --------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def cacf_in_windows(power, idx_start, idx_end):
    """Mean absolute autocorrelation (MCC) of every bin.

    Statistic: mean_{k=1..n-1} |r(k)|, r(k) = ACF(k) / ACF(0) of the centred power.

    Args:
        power: relative power.
        idx_start, idx_end: bin i covers power[idx_start[i]:idx_end[i]].

    Returns:
        MCC per bin (NaN for bins with < 2 points or zero variance).
    """
    n_windows = len(idx_start)
    results = np.full(n_windows, np.nan)
    for i in prange(n_windows):
        start, end = idx_start[i], idx_end[i]
        n = end - start
        if n < 2:
            continue
        x = power[start:end]
        x_centered = x - np.mean(x)
        nfft = scipy.fft.next_fast_len(2 * n - 1)               # zero-padded: linear ACF
        spectrum = scipy.fft.rfft(x_centered, n=nfft, workers=1)
        corr = scipy.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, workers=1)
        if corr[0] <= 0.0:                                      # constant window
            continue
        results[i] = np.mean(np.abs(corr[1:n])) / corr[0]
    return results


@njit(cache=True, parallel=True)
def cacf_delta_nu_in_windows(power, idx_start, idx_end, k_lo, k_hi):
    """Collapsed ACF ('delta_nu') of every bin and its white-noise mean and sd.

    Statistic: mean over the bands m of the band mean of r(k) = ACF(k) / ACF(0) over
    k_lo[i, m] <= k <= k_hi[i, m]; every band that fits in the bin has equal weight.
    White noise: E r(k) = -(n-k)/n^2, Var r(k) = (n-k)/n^2, lags conditioned on the sum
    rule sum_{k>=1} r(k) = -1/2 (Var S = sum a_k^2 Var r_k - mean^2 2n/(n-1)).

    Args:
        power: relative power.
        idx_start, idx_end: bin i covers power[idx_start[i]:idx_end[i]].
        k_lo, k_hi: (n_bins, n_bands) lag ranges in frequency points (not overlapping).

    Returns:
        (statistic, null_mean, null_sd) per bin; NaN where no band fits in the bin.
    """
    n_windows = len(idx_start)
    stat = np.full(n_windows, np.nan)
    null_mean = np.full(n_windows, np.nan)
    null_sd = np.full(n_windows, np.nan)
    for i in prange(n_windows):
        start, end = idx_start[i], idx_end[i]
        n = end - start
        if n < 3:
            continue
        x = power[start:end]
        x_centered = x - np.mean(x)
        nfft = scipy.fft.next_fast_len(2 * n - 1)
        spectrum = scipy.fft.rfft(x_centered, n=nfft, workers=1)
        corr = scipy.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, workers=1)
        c0 = corr[0]
        if not c0 > 0.0:
            continue
        total, mean0, var0, n_bands = 0.0, 0.0, 0.0, 0
        for m in range(k_lo.shape[1]):
            lo = max(1, k_lo[i, m])
            hi = min(n - 1, k_hi[i, m])
            if hi < lo:                                         # band beyond the bin
                continue
            b_tot, b_mean, b_var, count = 0.0, 0.0, 0.0, 0
            for k in range(lo, hi + 1):
                b_tot += corr[k] / c0
                b_mean -= (n - k) / (n * n)
                b_var += (n - k) / (n * n)
                count += 1
            total += b_tot / count                              # band means, equal weights
            mean0 += b_mean / count
            var0 += b_var / (count * count)
            n_bands += 1
        if n_bands > 0:
            stat[i] = total / n_bands
            null_mean[i] = mean0 / n_bands
            # sum rule sum_{k>=1} r(k) = -1/2: condition the lags on it
            var = var0 / (n_bands * n_bands) - null_mean[i] ** 2 * 2.0 * n / (n - 1)
            null_sd[i] = np.sqrt(var) if var > 0 else np.nan
    return stat, null_mean, null_sd


def delta_nu_lag_ranges(centres, frequency, window, multiples):
    """Lag bands within +- window x m x Delta-nu(bin centre) of m x Delta-nu.

    Bands that would overlap are trimmed so that no lag is in two bands.

    Args:
        centres: bin centres [muHz].
        frequency: frequency grid [muHz] (uniform).
        window: half-width as a fraction of m x Delta-nu.
        multiples: multiples m of Delta-nu (ascending).

    Returns:
        (k_lo, k_hi), each (n_bins, n_bands), in frequency points.
    """
    df = frequency[1] - frequency[0]
    dnu = delta_nu(centres)[:, None] * np.sort(np.asarray(multiples, dtype=float))[None, :]
    k_lo = np.ceil(dnu * (1 - window) / df).astype(np.int64)
    k_hi = np.floor(dnu * (1 + window) / df).astype(np.int64)
    k_hi[:, :-1] = np.minimum(k_hi[:, :-1], k_lo[:, 1:] - 1)      # no shared lags
    return np.ascontiguousarray(k_lo), np.ascontiguousarray(k_hi)


def acf_map(frequency, rel_power, centres, idx_start, idx_end, n_lag: int = 300,
            lag_unit: str = "dnu"):
    """2D ACF of every bin for plotting.

    Args:
        frequency: frequency grid [muHz].
        rel_power: relative power.
        centres, idx_start, idx_end: bins (see log_numax_bins).
        n_lag: points on the lag axis.
        lag_unit: "dnu" (lag / Delta-nu(bin centre)) or "muHz" (linear frequency lag).

    Returns:
        (lag grid in lag_unit, r(k) / white-noise sd [n_lag, n_bins]); NaN where a bin
        has no such lag.
    """
    df = frequency[1] - frequency[0]
    dnu = delta_nu(centres) if lag_unit == "dnu" else np.ones(len(centres))
    widths = (idx_end - idx_start) * df / dnu
    grid = np.linspace(0.0, np.nanmax(widths), n_lag)
    out = np.full((n_lag, len(centres)), np.nan)
    for i, (a, b) in enumerate(zip(idx_start, idx_end)):
        n = b - a
        if n < 3:
            continue
        x = rel_power[a:b] - np.mean(rel_power[a:b])
        nfft = scipy.fft.next_fast_len(2 * n - 1)
        corr = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:n]
        if not corr[0] > 0:
            continue
        k = np.arange(n)
        z = corr / corr[0] * n / np.sqrt(np.maximum(n - k, 1))
        kk = grid * dnu[i] / df
        ok = (kk >= 1) & (kk <= n - 1)
        out[ok, i] = np.interp(kk[ok], k, z)
    return grid, out


def relative_power(frequency, power, cfg, seed):
    """Relative power (P - median) / median.

    Args:
        frequency: frequency grid [muHz].
        power: PSD.
        cfg: ACFConfig (min_freq, replace_below_min_freq, median_filter_factor).
        seed: random seed for the white-noise replacement.

    Returns:
        (relative power, running median).
    """
    if cfg.replace_below_min_freq:
        power = replace_below_min_freq(frequency, power, cfg.min_freq, seed)
    med = running_median(frequency, power, cfg.median_filter_factor)
    return (power - med) / med, med


def calculate_cacf(frequency, rel_power, cfg) -> dict:
    """Raw and smoothed CACF for every grid setting.

    Args:
        frequency: frequency grid [muHz].
        rel_power: relative power.
        cfg: ACFConfig.

    Returns:
        {(overlap_scale, width_factor, smoothing_factor): {bin_centers, N_data_points_in_bin,
        cacf_raw, cacf_smoothed, noise_mean[, null_sd]}}; cacf_raw is the collapsed
        autocorrelation minus its white-noise mean (noise_mean).
    """
    t0 = time.perf_counter()
    frequency = np.ascontiguousarray(frequency)
    rel_power = np.ascontiguousarray(rel_power)
    min_freq = float(max(frequency[0], cfg.min_freq))
    results = {}
    for scale in as_float_list(cfg.overlap_scale)[::-1]:
        for factor in as_float_list(cfg.width_factor)[::-1]:
            centres, i0, i1 = log_numax_bins(frequency, min_freq, cfg.max_freq, factor, scale)
            if cfg.collapse == "all_lags":
                noise_mean = acf_noise_mean(i1 - i0)
                raw = cacf_in_windows(rel_power, i0, i1) - noise_mean
                extra = {"noise_mean": noise_mean}
            else:
                k_lo, k_hi = delta_nu_lag_ranges(centres, frequency, cfg.dnu_window,
                                                 cfg.dnu_multiples)
                stat, noise_mean, null_sd = cacf_delta_nu_in_windows(rel_power, i0, i1, k_lo, k_hi)
                raw = stat - noise_mean
                extra = {"noise_mean": noise_mean, "null_sd": null_sd}
            for sf, smoothed in smooth_over_envelope(centres, raw, cfg.smoothing_factor).items():
                results[(scale, factor, sf)] = {
                    "bin_centers": centres,
                    "N_data_points_in_bin": (i1 - i0).tolist(),
                    "cacf_raw": raw,
                    "cacf_smoothed": smoothed,
                    **extra,
                }
    logger.debug("CACF calculation: %.2f s", time.perf_counter() - t0)
    return results


# --------------------------------------------------------------------------------------
# Fitting (Enrico Corsaro, INAF-Catania, 2026)
# --------------------------------------------------------------------------------------
def fit_cacf(raw, smoothed, centres, fap, initial_numax, max_iterations, n_sigma, min_freq,
             at_numax=False):
    """Gaussian (with offset) fit to the smoothed CACF.

    Args:
        raw: raw CACF per bin (used for the significance test).
        smoothed: smoothed CACF per bin (fitted).
        centres: bin centres [muHz].
        fap: threshold per bin (or scalar); at least one raw bin must reach it.
        initial_numax: starting guess [muHz] or None (= highest smoothed bin).
        max_iterations: successive fits, each after removing the previous peak; the fit
            with the largest integral within n_sigma is kept.
        n_sigma: half-width of the integral window in fitted sigma.
        min_freq: lower bound for the fitted centre [muHz].
        at_numax: also require a significant bin within one envelope FWHM of the fit.

    Returns:
        (ufloat(numax, formal error), [amp, sigma, centre, offset]); NaN if no fit.
    """
    nan_result = ufloat(np.nan, np.nan), np.full(4, np.nan)
    x = np.asarray(centres, dtype=float)
    y_raw = np.asarray(raw, dtype=float)
    y_smooth = np.asarray(smoothed, dtype=float)
    fap = np.asarray(fap, dtype=float)
    if fap.ndim == 0:
        fap = np.full(x.shape, fap.item())

    significant = np.isfinite(x) & np.isfinite(y_raw) & np.isfinite(fap) & (y_raw >= fap)
    if not np.any(significant):
        return nan_result

    ok = np.isfinite(x) & np.isfinite(y_smooth)
    x_res, y_res = x[ok], y_smooth[ok]
    if x_res.size < 5:                       # 4 free parameters
        return nan_result

    best = (-np.inf, None, None)
    for it in range(int(max_iterations)):
        if x_res.size < 5:
            break
        try:
            numax0 = initial_numax if (it == 0 and initial_numax) else x_res[np.argmax(y_res)]
            numax0 = np.clip(numax0, x_res.min(), x_res.max())
            amp0 = 0.8 * np.max(y_res)
            fwhm0 = envelope_fwhm(numax0)
            p0 = [amp0, fwhm0 / FWHM_TO_SIGMA, numax0, np.median(y_res)]   # sigma, not FWHM
            lower = [0, 1e-6, min_freq, y_res.min()]
            upper = [1.5 * amp0, 2 * fwhm0, x_res.max(), y_res.max()]
            popt, pcov = curve_fit(gaussian, x_res, y_res, p0=p0, bounds=(lower, upper))

            window = np.abs(x_res - popt[2]) <= n_sigma * popt[1]
            integral = (trapezoid(y_res[window], x_res[window]) if window.any()
                        else trapezoid(y_res, x_res))
            if integral > best[0]:
                best = (integral, popt, pcov)
            x_res, y_res = x_res[~window], y_res[~window]
        except (RuntimeError, ValueError):
            break

    _, popt, pcov = best
    if popt is None:
        return nan_result
    if at_numax and not significant_near(x, y_raw, fap, popt[2]):
        return nan_result
    numax_unc = np.sqrt(np.diag(pcov))[2]               # formal error
    return ufloat(popt[2], abs(numax_unc)), popt


def fit_all(results, initial_numax, cfg) -> dict:
    """Fit every grid setting (in place).

    Args:
        results: output of calculate_cacf with "fap" added.
        initial_numax: starting guess [muHz] or None.
        cfg: ACFConfig.

    Returns:
        results, with fit_numax (ufloat) and fit_params per setting.
    """
    for data in results.values():
        numax, popt = fit_cacf(data["cacf_raw"], data["cacf_smoothed"], data["bin_centers"],
                               data["fap"], initial_numax, cfg.max_acf_fit_iterations,
                               cfg.n_sigma_numax_acf, cfg.min_freq, cfg.significance_at_numax)
        data["fit_numax"] = numax
        data["fit_params"] = {"amp": popt[0], "sigma": popt[1], "numax": popt[2],
                              "y": popt[3], "numax_err": numax.s}
    return results


# --------------------------------------------------------------------------------------
# Method class
# --------------------------------------------------------------------------------------
class NumaxFromACF:
    """Run the ACF method on one spectrum."""

    name = "ACF"
    info_folder = "ACF_info"

    def __init__(self, psd, acf_config, config, target="unknown", initial_numax=None):
        """Args:
            psd: PSDData.
            acf_config: ACFConfig.
            config: ProcessingConfig.
            target: star name (output folder).
            initial_numax: starting guess for the fits [muHz] or None.
        """
        self.frequency = psd.frequency
        self.power = psd.psd
        self.cfg = acf_config
        self.config = config
        self.id = target or "unknown"
        self.initial_numax = initial_numax

    def compute(self):
        """Relative power, CACF, thresholds and fits for every grid setting.

        Returns:
            self (results in self.results).
        """
        t0 = time.perf_counter()
        with numba_threads(self.config.n_threads):
            self.normalized_power, self.med_filter = relative_power(
                self.frequency, self.power, self.cfg, self.config.random_seed)
            results = calculate_cacf(self.frequency, self.normalized_power, self.cfg)
        results = add_faps(results, "ACF", self.cfg)
        self.results = fit_all(results, self.initial_numax, self.cfg)
        logger.info("%s: ACF done in %.1f s", self.id, time.perf_counter() - t0)
        return self

    @property
    def numax_estimate(self):
        """Median numax and median formal error over the grid (ufloat)."""
        return median_estimate(self.results)

    def _dir(self) -> Path:
        """Output folder <results_directory>/<target>/ACF_info."""
        return Path(self.config.results_directory) / str(self.id) / self.info_folder

    def save_all_data(self):
        """Save all results as ACF_info/all_data.pkl.gz."""
        save_pickle(self.results, self._dir() / "all_data.pkl.gz")

    def save_numax_estimates(self):
        """Save numax per grid setting as ACF_info/numax_estimates.txt."""
        save_numax_estimates(self.results, self._dir() / "numax_estimates.txt")

    def plot(self):
        """Save ACF.png (spectrum, optional 2D ACF, CACF curves and fits)."""
        from ..plotting import plot_acf
        plot_acf(self)
