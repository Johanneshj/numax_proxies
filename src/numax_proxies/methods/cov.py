"""CoV method: numax from the coefficient of variation (std / mean) of P / median in bins
(Bell+2019, Viani+2019).

For every grid setting: bin P / median (width w x Delta-nu), compute the CoV per bin, smooth,
start at the highest significant bin and fit a Gaussian (offset 1). numax = median over the grid.
"""
import logging
import time
import warnings
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import OptimizeWarning, curve_fit
from uncertainties import ufloat

from ..binning import (as_float_list, log_numax_bins, replace_below_min_freq, running_median,
                       smooth_over_envelope)
from ..fitting import gaussian_offset_one, gaussian_offset_one_jac
from ..seismic import cov_upper_limit, envelope_fwhm
from ..significance import add_faps, significant_near
from ..threads import numba_threads
from .common import correct_fits, median_estimate, save_numax_estimates, save_pickle

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# CoV per bin (prefix sums: O(1) per bin)
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _prefix_sums(power):
    """Cumulative sums of power and power^2, and the count of non-finite values."""
    n = len(power)
    s = np.zeros(n + 1)
    sq = np.zeros(n + 1)
    bad = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        v = power[i]
        s[i + 1], sq[i + 1], bad[i + 1] = s[i], sq[i], bad[i]
        if np.isfinite(v):
            s[i + 1] += v
            sq[i + 1] += v * v
        else:
            bad[i + 1] += 1
    return s, sq, bad


@njit(cache=True)
def _cov_two_pass(power, start, end):
    """CoV of one bin with a two-pass variance (precision fallback); 1 if undefined."""
    n = end - start
    if n <= 1:
        return 1.0
    total = 0.0
    for i in range(start, end):
        if not np.isfinite(power[i]):
            return 1.0
        total += power[i]
    mean = total / n
    if not np.isfinite(mean) or mean == 0.0:
        return 1.0
    ss = 0.0
    for i in range(start, end):
        d = power[i] - mean
        ss += d * d
    std = np.sqrt(ss / (n - 1))
    return std / mean if np.isfinite(std) else 1.0


@njit(cache=True)
def cov_in_windows(power, idx_start, idx_end, s, sq, bad):
    """CoV (std / mean) of every bin; 1 for empty or undefined bins."""
    eps = 2.220446049250313e-16
    out = np.empty(len(idx_start))
    for i in range(len(idx_start)):
        start, end = idx_start[i], idx_end[i]
        n = end - start
        if n <= 1 or bad[end] - bad[start] > 0:
            out[i] = 1.0
            continue
        total = s[end] - s[start]
        total_sq = sq[end] - sq[start]
        mean = total / n
        if not np.isfinite(mean) or mean == 0.0:
            out[i] = 1.0
            continue
        if not np.isfinite(total) or not np.isfinite(total_sq):
            out[i] = _cov_two_pass(power, start, end)
            continue
        var_num = total_sq - total * mean
        scale = max(abs(total_sq), abs(total * mean))
        if not np.isfinite(var_num) or var_num <= 64.0 * eps * scale:
            out[i] = _cov_two_pass(power, start, end)
            continue
        std = np.sqrt(var_num / (n - 1))
        out[i] = std / mean if np.isfinite(std) else 1.0
    return out


def normalized_power(frequency, power, cfg, seed, med=None):
    """Normalised power P / median and the running median."""
    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    power = np.ascontiguousarray(power, dtype=np.float64)
    if cfg.replace_below_min_freq:
        power = replace_below_min_freq(frequency, power, cfg.min_freq, seed)
    if med is None or cfg.replace_below_min_freq:
        med = running_median(frequency, power, cfg.median_filter_factor)
    return power / med, med


def calculate_covs(frequency, norm_power, cfg) -> dict:
    """Raw and smoothed CoV for every grid setting."""
    t0 = time.perf_counter()
    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    if np.any(np.diff(frequency) < 0):
        raise ValueError("Frequency must be sorted.")
    s, sq, bad = _prefix_sums(norm_power)
    min_freq = float(max(frequency[0], cfg.min_freq))
    results = {}
    for scale in as_float_list(cfg.overlap_scale):
        for factor in as_float_list(cfg.width_factor):
            centres, i0, i1 = log_numax_bins(frequency, min_freq, cfg.max_freq, factor, scale)
            raw = cov_in_windows(norm_power, i0, i1, s, sq, bad)
            for sf, smoothed in smooth_over_envelope(centres, raw, cfg.smoothing_factor).items():
                results[(scale, factor, sf)] = {
                    "bin_centers": centres,
                    "N_data_points_in_bin": (i1 - i0).tolist(),
                    "cov": raw,
                    "cov_smoothed": smoothed,
                }
    logger.debug("CoV calculation: %.2f s", time.perf_counter() - t0)
    return results


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
def initial_numax_from_cov(data) -> float:
    """Starting guess: the highest smoothed CoV among the significant bins (NaN if none)."""
    bcs = np.asarray(data["bin_centers"], dtype=float)
    cov = np.asarray(data["cov"], dtype=float)
    scov = np.asarray(data["cov_smoothed"], dtype=float)
    valid = np.isfinite(bcs) & np.isfinite(cov) & np.isfinite(scov) & (bcs > 0)
    if data.get("fap") is not None:
        fap = np.broadcast_to(np.asarray(data["fap"], dtype=float), cov.shape)
        valid &= np.isfinite(fap) & (cov > fap)
    if not np.any(valid):
        return np.nan
    below = valid & (cov < cov_upper_limit(bcs))
    candidates = np.flatnonzero(below if np.any(below) else valid)
    return float(bcs[candidates[np.argmax(scov[candidates])]])


def _peak_cov(x, y, centre, half_width) -> float:
    """Highest CoV within +- half_width of centre."""
    inside = np.abs(x - centre) <= half_width
    return float(np.max(y[inside])) if np.any(inside) else float(y[np.argmin(np.abs(x - centre))])


def fit_gaussian(x, y, numax_init):
    """Gaussian (offset 1) fit to the smoothed CoV: (numax, params, status = fit | rejected | failed)."""
    nan_result = ufloat(np.nan, np.nan), np.full(3, np.nan), "failed"
    rejected = ufloat(np.nan, np.nan), np.full(3, np.nan), "rejected"
    if numax_init is None or not np.isfinite(numax_init) or numax_init <= 0:
        return nan_result
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if len(x) < 3:
        return nan_result
    try:
        lo, hi = x.min(), x.max()                       # centre within the fitted bins
        # model 1 + A g: the excess above 1 is the amplitude; A >= 0 (an envelope is a peak,
        # never a dip in the CoV)
        p0 = [max(np.max(y) - 1.0, 1e-3), envelope_fwhm(numax_init), np.clip(numax_init, lo, hi)]
        bounds = ([0.0, 1e-6, lo], [np.inf, np.inf, hi])
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(gaussian_offset_one, x, y, p0=p0, bounds=bounds,
                                   jac=gaussian_offset_one_jac)
        sigma, centre = abs(popt[1]), popt[2]
        centre_unc = np.sqrt(np.diag(pcov))[2]          # formal error
        if not np.all(np.isfinite(popt)) or sigma <= 0:
            raise ValueError("non-finite or invalid parameters")
        if np.isclose(centre, lo, rtol=1e-6) or np.isclose(centre, hi, rtol=1e-6):
            return rejected                             # peak outside the window
        peak = _peak_cov(x, y, centre, sigma)
        if peak > cov_upper_limit(centre):
            logger.debug("Rejected fit at %.2f muHz: peak CoV %.3f above limit", centre, peak)
            return rejected
        return ufloat(centre, centre_unc), popt, "fit"
    except (RuntimeError, OptimizeWarning, ValueError, TypeError, FloatingPointError,
            OverflowError) as exc:
        logger.debug("Gaussian fit failed at initial numax %.2f muHz: %s", numax_init, exc)
        return nan_result


def weighted_mean_numax(x, y):
    """CoV-weighted mean frequency (fallback when no Gaussian fits)."""
    denominator = np.sum(y)
    if not np.isfinite(denominator) or denominator == 0:
        return ufloat(np.nan, np.nan)
    estimate = np.sum(x * y) / denominator
    if not np.isfinite(estimate):
        return ufloat(np.nan, np.nan)
    return ufloat(estimate, np.std(x, ddof=1) if len(x) > 1 else np.nan)


MAX_COMB_RETRIES = 3          # refits after a fit fails the comb check


def _mask_near(bcs, values, centres):
    """Copy of values with NaN within one envelope FWHM of each centre."""
    out = np.array(values, dtype=float, copy=True)
    for c in centres:
        out[np.abs(bcs - c) <= envelope_fwhm(c)] = np.nan
    return out


def _fit_one(data, initial_numax, use_fallback, at_numax):
    """Fit one grid setting: (numax, params, fit_type)."""
    nan_params = {"amp": np.nan, "sigma": np.nan, "numax": np.nan, "numax_err": np.nan}
    none = ufloat(np.nan, np.nan), dict(nan_params), "none"
    numax_init = initial_numax if initial_numax is not None else initial_numax_from_cov(data)
    if numax_init is None or not np.isscalar(numax_init) or not np.isfinite(numax_init) \
            or numax_init <= 0:
        return none
    bcs = np.asarray(data["bin_centers"], dtype=float)
    scov = np.asarray(data["cov_smoothed"], dtype=float)
    width = envelope_fwhm(numax_init)
    window = (np.isfinite(bcs) & np.isfinite(scov)
              & (bcs >= numax_init - width) & (bcs <= numax_init + width))
    if not np.any(window):
        return none
    x, y = bcs[window], scov[window]
    result = none
    numax, popt, status = fit_gaussian(x, y, numax_init)
    if status == "fit":
        result = numax, {"amp": popt[0], "sigma": popt[1], "numax": popt[2],
                         "numax_err": numax.s}, "gaussian"
    elif status == "rejected":
        result = ufloat(np.nan, np.nan), dict(nan_params), "rejected"
    elif use_fallback:
        fallback = weighted_mean_numax(x, y)
        if np.isfinite(fallback.n):
            result = fallback, dict(nan_params), "fallback"
    if at_numax and result[2] in ("gaussian", "fallback") and not significant_near(
            bcs, data["cov"], data["fap"], result[0].n):
        result = ufloat(np.nan, np.nan), dict(nan_params), "not_significant"
    return result


def fit_all(results, initial_numax=None, use_fallback=True, at_numax=False,
            comb_checker=None, exclude=()) -> dict:
    """Fit every grid setting (in place).

    A fit that fails the comb check is redone without that region; bins near the values in
    exclude are left out (comb-scan rejections).
    """
    counts = dict.fromkeys(("gaussian", "fallback", "rejected", "not_significant", "no_comb",
                            "none"), 0)
    for data in results.values():
        bcs = np.asarray(data["bin_centers"], dtype=float)
        def masked(centres):
            if not len(centres):
                return data
            return dict(data, cov=_mask_near(bcs, data["cov"], centres),
                        cov_smoothed=_mask_near(bcs, data["cov_smoothed"], centres))

        excluded = []
        numax, params, fit_type = _fit_one(masked(exclude), None if len(exclude) else initial_numax,
                                           use_fallback, at_numax)
        while comb_checker is not None and fit_type in ("gaussian", "fallback") \
                and not comb_checker(numax.n):
            excluded.append(numax.n)
            if initial_numax is not None or len(excluded) > MAX_COMB_RETRIES:
                numax, params, fit_type = ufloat(np.nan, np.nan), {}, "no_comb"
                break
            numax, params, fit_type = _fit_one(masked([*exclude, *excluded]), None,
                                               use_fallback, at_numax)
        if excluded and fit_type in ("none", "not_significant", "rejected"):
            fit_type = "no_comb"                         # nothing left after the rejections
        nan_params = {"amp": np.nan, "sigma": np.nan, "numax": np.nan, "numax_err": np.nan}
        data["fit_numax"], data["fit_params"], data["fit_type"] = numax, params or nan_params, fit_type
        data["comb_rejected"] = excluded
        counts[fit_type] += 1
    logger.info("CoV fits: %(gaussian)d Gaussian, %(fallback)d fallback, %(rejected)d rejected, "
                "%(not_significant)d not significant at numax, %(no_comb)d no comb, %(none)d none",
                counts)
    return results


# --------------------------------------------------------------------------------------
# Method class
# --------------------------------------------------------------------------------------
class NumaxFromCoV:
    """Run the CoV method on one spectrum."""

    name = "CoV"
    info_folder = "CoV_info"

    def __init__(self, psd, cov_config, config, target="unknown", initial_numax=None):
        """Set up the method; the pipeline fills in median, comb_checker and background_median."""
        self.frequency = psd.frequency
        self.power = psd.psd
        self.cfg = cov_config
        self.config = config
        self.id = target or "unknown"
        self.initial_numax = initial_numax
        self.comb_check_mode = None          # "near_artefacts" | "always" (set by the pipeline)
        self.comb_checker = None             # shared artefacts.CombChecker (set by the pipeline)
        self.median = None                   # running median of psd (set by the pipeline)
        self.background_median = None        # set -> background-corrected numax per fit

    def compute(self):
        """Normalised power, CoV, thresholds and fits for every grid setting."""
        t0 = time.perf_counter()
        with numba_threads(self.config.n_threads):
            self.normalized_power, self.med_filter = normalized_power(
                self.frequency, self.power, self.cfg, self.config.random_seed,
                self.median)
            results = calculate_covs(self.frequency, self.normalized_power, self.cfg)
        results = add_faps(results, "CoV", self.cfg)
        checker = self.comb_checker if self.comb_check_mode else None
        if self.comb_check_mode and checker is None:
            from ..artefacts import CombChecker
            checker = CombChecker(self.frequency, self.normalized_power - 1.0,
                                  self.comb_check_mode, self.config.comb_check_tolerance,
                                  self.config.comb_check_fap,
                                  self.config.comb_check_distance)
        self._checker = checker
        self.results = fit_all(results, self.initial_numax, self.cfg.use_fallback,
                               self.cfg.significance_at_numax, checker)
        self._correct_background()
        logger.info("%s: CoV done in %.1f s", self.id, time.perf_counter() - t0)
        return self

    def refit(self, exclude):
        """Repeat the fits without the bins near the numax values in exclude."""
        self.results = fit_all(self.results, self.initial_numax, self.cfg.use_fallback,
                               self.cfg.significance_at_numax, self._checker, list(exclude))
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
        """Output folder <results_directory>/<target>/CoV_info."""
        return Path(self.config.results_directory) / str(self.id) / self.info_folder

    def save_all_data(self):
        """Save all results as all_data.pkl.gz."""
        save_pickle(self.results, self._dir() / "all_data.pkl.gz")

    def save_numax_estimates(self):
        """Save numax per grid setting as numax_estimates.txt."""
        save_numax_estimates(self.results, self._dir() / "numax_estimates.txt")

    def plot(self):
        """Save CoVs.png (curves, fits and the CoV upper limit)."""
        from ..plotting import plot_cov
        plot_cov(self)
