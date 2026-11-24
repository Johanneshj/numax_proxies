"""ACF method: numax from the collapsed 2D autocorrelation function (Huber+2009, Viani+2019).

For every grid setting: bin the relative power (width w x Delta-nu), collapse each bin's ACF to
one number minus its white-noise mean (ACF_CONFIG.collapse: all_lags, delta_nu, or
autocovariance for the ACV method), smooth, test against the per-bin threshold and fit a
Gaussian. numax = median over the grid.
"""
import dataclasses
import logging
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
from ..fitting import FWHM_TO_SIGMA, gaussian, gaussian_jac
from ..seismic import delta_nu, envelope_fwhm
from ..significance import acf_noise_mean, add_faps, significant_near
from ..threads import numba_threads
from .common import correct_fits, median_estimate, save_numax_estimates, save_pickle

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Collapsed ACF
# --------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def cacf_in_windows(power, idx_start, idx_end):
    """Mean |r(k)| over all lags (MCC) of every bin; NaN for empty or constant bins."""
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
def acv_in_windows(power, idx_start, idx_end):
    """Mean |ACF(k)| / n over all lags of every bin (the ACV method: not divided by ACF(0))."""
    n_windows = len(idx_start)
    results = np.full(n_windows, np.nan)
    for i in prange(n_windows):
        start, end = idx_start[i], idx_end[i]
        n = end - start
        if n < 2:
            continue
        x = power[start:end]
        x_centered = x - np.mean(x)
        nfft = scipy.fft.next_fast_len(2 * n - 1)
        spectrum = scipy.fft.rfft(x_centered, n=nfft, workers=1)
        corr = scipy.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, workers=1)
        results[i] = np.mean(np.abs(corr[1:n])) / n
    return results


@njit(cache=True, parallel=True)
def cacf_delta_nu_in_windows(power, idx_start, idx_end, k_lo, k_hi):
    """Mean r(k) in the lag bands of every bin, with its white-noise mean and sd: (stat, mean, sd)."""
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


def band_edges(multiples, window, mode="relative"):
    """Edges of the lag bands in units of Delta-nu: (lo, hi), before trimming of overlaps."""
    m = np.sort(np.asarray(multiples, dtype=float))
    if mode == "fixed":
        return m - window, m + window
    return m * (1 - window), m * (1 + window)


def delta_nu_lag_ranges(centres, frequency, window, multiples, mode="relative"):
    """Lag bands around m x Delta-nu(bin centre) in points, trimmed so they do not overlap: (k_lo, k_hi)."""
    df = frequency[1] - frequency[0]
    lo, hi = band_edges(multiples, window, mode)
    dnu = delta_nu(centres)[:, None]
    k_lo = np.ceil(dnu * lo[None, :] / df).astype(np.int64)
    k_hi = np.floor(dnu * hi[None, :] / df).astype(np.int64)
    k_hi[:, :-1] = np.minimum(k_hi[:, :-1], k_lo[:, 1:] - 1)      # no shared lags
    return np.ascontiguousarray(k_lo), np.ascontiguousarray(k_hi)


def acf_map(frequency, rel_power, centres, idx_start, idx_end, n_lag: int = 300,
            lag_unit: str = "dnu", normalise: bool = True):
    """2D ACF of every bin for plotting: r(k) / white-noise sd, or ACF(k) / n with normalise=False. Returns (lags, map)."""
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
        z = corr / corr[0] * n / np.sqrt(np.maximum(n - k, 1)) if normalise else corr / n
        kk = grid * dnu[i] / df
        ok = (kk >= 1) & (kk <= n - 1)
        out[ok, i] = np.interp(kk[ok], k, z)
    return grid, out


def relative_power(frequency, power, cfg, seed, med=None):
    """Relative power (P - median) / median and the running median."""
    if cfg.replace_below_min_freq:
        power = replace_below_min_freq(frequency, power, cfg.min_freq, seed)
    if med is None or cfg.replace_below_min_freq:
        med = running_median(frequency, power, cfg.median_filter_factor)
    return (power - med) / med, med


def calculate_cacf(frequency, rel_power, cfg) -> dict:
    """Raw and smoothed collapsed statistic for every grid setting."""
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
            elif cfg.collapse == "autocovariance":
                noise_mean = acf_noise_mean(i1 - i0, "ACV")
                raw = acv_in_windows(rel_power, i0, i1) - noise_mean
                extra = {"noise_mean": noise_mean}
            else:
                k_lo, k_hi = delta_nu_lag_ranges(centres, frequency, cfg.dnu_window,
                                                 cfg.dnu_multiples, cfg.dnu_window_mode)
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
    """Gaussian fit to the smoothed curve of one grid setting.

    Needs at least one raw bin above the threshold (near the fit with at_numax). With
    max_iterations > 1 the peak is removed and refitted; the fit with the largest integral is
    kept. Returns (ufloat numax, [amp, sigma, centre, offset]), NaN if no fit.
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
            popt, pcov = curve_fit(gaussian, x_res, y_res, p0=p0, bounds=(lower, upper),
                                   jac=gaussian_jac)

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


MAX_COMB_RETRIES = 3          # refits after a fit fails the comb check


def fit_all(results, initial_numax, cfg, comb_checker=None, exclude=()) -> dict:
    """Fit every grid setting (in place).

    A fit that fails the comb check is redone without that region; bins near the values in
    exclude are left out (comb-scan rejections).
    """
    def fit(raw, smoothed, guess):
        return fit_cacf(raw, smoothed, data["bin_centers"], data["fap"], guess,
                        cfg.max_acf_fit_iterations, cfg.n_sigma_numax_acf, cfg.min_freq,
                        cfg.significance_at_numax)

    for data in results.values():
        bcs = np.asarray(data["bin_centers"], dtype=float)
        raw = np.asarray(data["cacf_raw"], dtype=float)
        smoothed = np.asarray(data["cacf_smoothed"], dtype=float)
        def masked(centres):
            mask = np.zeros(bcs.size, dtype=bool)
            for c in centres:
                mask |= np.abs(bcs - c) <= envelope_fwhm(c)
            return np.where(mask, np.nan, raw), np.where(mask, np.nan, smoothed)

        excluded = []
        numax, popt = fit(*masked(exclude), None if len(exclude) else initial_numax)
        while comb_checker is not None and np.isfinite(numax.n) and not comb_checker(numax.n):
            excluded.append(numax.n)
            if initial_numax is not None or len(excluded) > MAX_COMB_RETRIES:
                numax, popt = ufloat(np.nan, np.nan), np.full(4, np.nan)
                break
            numax, popt = fit(*masked([*exclude, *excluded]), None)
        data["fit_numax"] = numax
        data["fit_params"] = {"amp": popt[0], "sigma": popt[1], "numax": popt[2],
                              "y": popt[3], "numax_err": numax.s}
        data["comb_rejected"] = excluded
    return results


# --------------------------------------------------------------------------------------
# Method class
# --------------------------------------------------------------------------------------
class NumaxFromACF:
    """Run the ACF method on one spectrum."""

    name = "ACF"
    info_folder = "ACF_info"

    def __init__(self, psd, acf_config, config, target="unknown", initial_numax=None):
        """Set up the method; the pipeline fills in median, comb_checker and background_median."""
        self.frequency = psd.frequency
        self.power = psd.psd
        self.cfg = acf_config
        self.config = config
        self.id = target or "unknown"
        self.initial_numax = initial_numax
        self.comb_check_mode = None          # "near_artefacts" | "always" (set by the pipeline)
        self.comb_checker = None             # shared artefacts.CombChecker (set by the pipeline)
        self.median = None                   # running median of psd (set by the pipeline)
        self.background_median = None        # set -> background-corrected numax per fit

    def compute(self):
        """Relative power, collapsed statistic, thresholds and fits for every grid setting."""
        t0 = time.perf_counter()
        with numba_threads(self.config.n_threads):
            self.normalized_power, self.med_filter = relative_power(
                self.frequency, self.power, self.cfg, self.config.random_seed,
                self.median)
            results = calculate_cacf(self.frequency, self.normalized_power, self.cfg)
        results = add_faps(results, self.name, self.cfg)
        checker = self.comb_checker if self.comb_check_mode else None
        if self.comb_check_mode and checker is None:
            from ..artefacts import CombChecker
            checker = CombChecker(self.frequency, self.normalized_power,
                                  self.comb_check_mode, self.config.comb_check_tolerance,
                                  self.config.comb_check_fap,
                                  self.config.comb_check_distance)
        self._checker = checker
        self.results = fit_all(results, self.initial_numax, self.cfg, checker)
        self._correct_background()
        logger.info("%s: %s done in %.1f s", self.id, self.name, time.perf_counter() - t0)
        return self

    def refit(self, exclude):
        """Repeat the fits without the bins near the numax values in exclude."""
        self.results = fit_all(self.results, self.initial_numax, self.cfg, self._checker,
                               list(exclude))
        self._correct_background()
        return self

    def _correct_background(self):
        """Background-correct every fitted numax if background_median is set."""
        if self.background_median is not None:
            correct_fits(self.results, self.frequency, self.background_median)

    @property
    def numax_estimate(self):
        """Median numax over the grid, or the comb-scan numax if one was set."""
        if getattr(self, "numax_override", None) is not None:
            return self.numax_override
        return median_estimate(self.results)

    def _dir(self) -> Path:
        """Output folder <results_directory>/<target>/<info_folder>."""
        return Path(self.config.results_directory) / str(self.id) / self.info_folder

    def save_all_data(self):
        """Save all results as all_data.pkl.gz."""
        save_pickle(self.results, self._dir() / "all_data.pkl.gz")

    def save_numax_estimates(self):
        """Save numax per grid setting as numax_estimates.txt."""
        save_numax_estimates(self.results, self._dir() / "numax_estimates.txt")

    def plot(self):
        """Save <name>.png (spectrum, optional 2D ACF, curves and fits)."""
        from ..plotting import plot_acf
        plot_acf(self)


class NumaxFromACV(NumaxFromACF):
    """The ACV method: the ACF method with the autocovariance collapse (ACF not divided by ACF(0))."""

    name = "ACV"
    info_folder = "ACV_info"

    def __init__(self, psd, acf_config, config, target="unknown", initial_numax=None):
        super().__init__(psd, dataclasses.replace(acf_config, collapse="autocovariance"),
                         config, target, initial_numax)
