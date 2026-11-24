"""Spectrum preparation shared by the methods: running median, bins of width w x Delta-nu, and smoothing of binned curves."""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .seismic import DNU_COEFF, DNU_EXP, ENV_COEFF, ENV_EXP, delta_nu, envelope_fwhm

DEFAULT_MEDIAN_FILTER_FACTOR = 2.0 * np.log(2)   # window = 2 ln2 x envelope FWHM


# --------------------------------------------------------------------------------------
# 1. White-noise replacement below min_freq
# --------------------------------------------------------------------------------------
def replace_below_min_freq(frequency: NDArray, power: NDArray, min_freq: float,
                           seed: int = 42) -> NDArray:
    """Copy of the PSD with the part below min_freq replaced by white noise."""
    power = np.array(power, dtype=np.float64, copy=True)
    mask = frequency <= min_freq
    if np.any(mask):
        rng = np.random.default_rng(seed)
        scale = np.median(power[mask]) / np.log(2)
        power[mask] = rng.exponential(scale=scale, size=mask.sum())
    return power


# --------------------------------------------------------------------------------------
# 2. Running pseudo-median
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _eval_indices(frequency, factor, step_fraction):
    """Points where the running median is evaluated (every step_fraction x window width)."""
    n = len(frequency)
    df = (frequency[-1] - frequency[0]) / (n - 1)
    out = np.empty(n, dtype=np.int64)
    m, i = 0, 0
    while i < n - 1:
        out[m] = i
        m += 1
        window = factor * ENV_COEFF * max(frequency[i], 0.0) ** ENV_EXP / df   # [points]
        i += max(1, int(step_fraction * window))
    out[m] = n - 1
    return out[:m + 1]


@njit(parallel=True, cache=True)
def _running_median_coarse(frequency, power, factor, eval_indices):
    """Median over factor x envelope FWHM at the given points."""
    n = len(power)
    df = (frequency[-1] - frequency[0]) / (n - 1)
    m = len(eval_indices)
    med_sub = np.empty(m)
    for j in prange(m):
        i = eval_indices[j]
        half_w = 0.5 * factor * ENV_COEFF * frequency[i] ** ENV_EXP
        start_idx = max(0, int(np.floor((frequency[i] - half_w - frequency[0]) / df)))
        end_idx = min(n, int(np.ceil((frequency[i] + half_w - frequency[0]) / df)))
        if end_idx > start_idx:
            med_sub[j] = np.median(power[start_idx:end_idx])
        else:
            med_sub[j] = power[i]
    return med_sub


def running_median(frequency: NDArray, power: NDArray,
                   factor: float = DEFAULT_MEDIAN_FILTER_FACTOR,
                   step_fraction: float = 0.01) -> NDArray:
    """Running median of the PSD over factor x envelope FWHM, evaluated in steps and interpolated."""
    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    power = np.ascontiguousarray(power, dtype=np.float64)
    if len(frequency) < 2:
        raise ValueError("At least two frequency points are required.")
    if not step_fraction > 0:
        raise ValueError("step_fraction must be positive.")
    idx = _eval_indices(frequency, factor, step_fraction)
    med_sub = _running_median_coarse(frequency, power, factor, idx)
    return np.interp(frequency, frequency[idx], med_sub)


# --------------------------------------------------------------------------------------
# 3. Bins of width  w * Delta-nu(nu)
# --------------------------------------------------------------------------------------
def log_numax_bins(frequency: NDArray, min_freq: float, max_freq: float | None,
                   width_factor: float, overlap_scale: float):
    """Overlapping bins of width width_factor x Delta-nu: (centres, idx_start, idx_end)."""
    if max_freq is None:
        max_freq = frequency[-1]

    centres = [float(max(frequency[0], min_freq))]
    while centres[-1] < max_freq:
        centre = centres[-1]
        next_centre = centre + (width_factor * DNU_COEFF * centre ** DNU_EXP) / overlap_scale
        if next_centre > max_freq:
            break
        centres.append(next_centre)
    centres = np.asarray(centres, dtype=np.float64)

    widths = delta_nu(centres, width_factor)
    idx_start = np.searchsorted(frequency, centres - widths / 2, side="left")
    idx_end = np.searchsorted(frequency, centres + widths / 2, side="right")
    return centres, idx_start, idx_end


# --------------------------------------------------------------------------------------
# 4. Smoothing over the envelope width
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _lower_bound(arr, value):
    """First index with arr[i] >= value."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound(arr, value):
    """First index with arr[i] > value."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _smooth_numba(centres, values, factor):
    """Running mean of values over +- factor x FWHM / 2 around each centre (NaNs ignored)."""
    n = len(centres)
    smoothed = np.full(n, np.nan)

    prefix_sum = np.zeros(n + 1)
    prefix_count = np.zeros(n + 1, dtype=np.int64)
    prefix_posinf = np.zeros(n + 1, dtype=np.int64)
    prefix_neginf = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        v = values[i]
        prefix_sum[i + 1] = prefix_sum[i]
        prefix_count[i + 1] = prefix_count[i]
        prefix_posinf[i + 1] = prefix_posinf[i]
        prefix_neginf[i + 1] = prefix_neginf[i]
        if np.isnan(v):
            continue
        elif np.isinf(v):
            if v > 0:
                prefix_posinf[i + 1] += 1
            else:
                prefix_neginf[i + 1] += 1
        else:
            prefix_sum[i + 1] += v
            prefix_count[i + 1] += 1

    for i in range(n):
        half_width = factor * (ENV_COEFF * centres[i] ** ENV_EXP) / 2.0
        start = _lower_bound(centres, centres[i] - half_width)
        end = _upper_bound(centres, centres[i] + half_width)
        if start >= end:
            continue
        n_pos = prefix_posinf[end] - prefix_posinf[start]
        n_neg = prefix_neginf[end] - prefix_neginf[start]
        if n_pos > 0 and n_neg > 0:
            continue
        if n_pos > 0:
            smoothed[i] = np.inf
            continue
        if n_neg > 0:
            smoothed[i] = -np.inf
            continue
        count = prefix_count[end] - prefix_count[start]
        if count > 0:
            smoothed[i] = (prefix_sum[end] - prefix_sum[start]) / count
    return smoothed


def smooth_over_envelope(centres: NDArray, values: NDArray, smoothing_factors) -> dict:
    """Running mean of a binned curve over each smoothing factor x envelope FWHM: {factor: curve}."""
    centres = np.ascontiguousarray(centres, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    if len(centres) != len(values):
        raise ValueError("centres and values must have equal lengths.")
    if np.any(np.diff(centres) < 0):
        raise ValueError("centres must be sorted in ascending order.")
    return {float(s): _smooth_numba(centres, values, float(s)) for s in as_float_list(smoothing_factors)}


def as_float_list(value) -> list[float]:
    """Scalar or list -> list of floats."""
    if np.ndim(value) == 0:
        return [float(value)]
    return [float(v) for v in value]


__all__ = ["replace_below_min_freq", "running_median", "log_numax_bins",
           "smooth_over_envelope", "as_float_list", "envelope_fwhm",
           "DEFAULT_MEDIAN_FILTER_FACTOR"]
