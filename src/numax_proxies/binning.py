"""Spectrum preparation shared by the ACF and CoV methods.

1. replace_below_min_freq  - optional: replace the PSD below min_freq with white noise
2. running_median          - pseudo-running median over an envelope-sized window
3. log_numax_bins          - overlapping bins of width  w * Delta-nu(nu)
4. smooth_over_envelope    - running mean over  s * envelope FWHM(nu)  (in bin-centre space)
"""
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
    """Replace the PSD at frequencies <= min_freq with chi^2_2 white noise.

    Args:
        frequency: frequency grid [muHz].
        power: PSD.
        min_freq: upper limit of the replaced range [muHz].
        seed: random seed.

    Returns:
        Copy of power; the noise mean is median / ln 2 of the replaced points.
    """
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
@njit(parallel=True, cache=True)
def _running_median_coarse(frequency, power, factor, stride):
    """Median in windows of factor x envelope FWHM, every stride points.

    Returns:
        (indices of the evaluated points, medians at those points).
    """
    n = len(power)
    df = (frequency[-1] - frequency[0]) / (n - 1)

    eval_indices = np.arange(0, n, stride)
    if eval_indices[-1] != n - 1:
        eval_indices = np.append(eval_indices, n - 1)

    m = len(eval_indices)
    med_sub = np.empty(m)
    env_width = factor * ENV_COEFF * (frequency[eval_indices] ** ENV_EXP)
    half_w = env_width / 2.0

    for j in prange(m):
        i = eval_indices[j]
        left_val = frequency[i] - half_w[j]
        right_val = frequency[i] + half_w[j]
        start_idx = max(0, int(np.floor((left_val - frequency[0]) / df)))
        end_idx = min(n, int(np.ceil((right_val - frequency[0]) / df)))
        window_data = power[start_idx:end_idx]
        if len(window_data) > 0:
            med_sub[j] = np.median(window_data)
        else:
            med_sub[j] = power[i]
    return eval_indices, med_sub


def running_median(frequency: NDArray, power: NDArray,
                   factor: float = DEFAULT_MEDIAN_FILTER_FACTOR, stride: int = 10) -> NDArray:
    """Running median of the PSD over factor x envelope FWHM(nu).

    Args:
        frequency: uniform frequency grid [muHz].
        power: PSD.
        factor: window width in envelope FWHM.
        stride: evaluate every stride points and interpolate in between.

    Returns:
        Running median on the full frequency grid.
    """
    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    power = np.ascontiguousarray(power, dtype=np.float64)
    if len(frequency) < 2:
        raise ValueError("At least two frequency points are required.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    eval_indices, med_sub = _running_median_coarse(frequency, power, factor, stride)
    return np.interp(frequency, frequency[eval_indices], med_sub)


# --------------------------------------------------------------------------------------
# 3. Bins of width  w * Delta-nu(nu)
# --------------------------------------------------------------------------------------
def log_numax_bins(frequency: NDArray, min_freq: float, max_freq: float | None,
                   width_factor: float, overlap_scale: float):
    """Overlapping bins of width width_factor x Delta-nu(bin centre).

    Args:
        frequency: frequency grid [muHz], ascending.
        min_freq: first bin centre [muHz].
        max_freq: last allowed bin centre [muHz]; None = end of the spectrum.
        width_factor: bin width in units of Delta-nu.
        overlap_scale: bins per bin width (step = width / overlap_scale).

    Returns:
        (centres, idx_start, idx_end); bin i covers power[idx_start[i]:idx_end[i]].
    """
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
    """First index with arr[i] >= value (arr ascending)."""
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
    """First index with arr[i] > value (arr ascending)."""
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
    """Mean of values over bins within +- factor x FWHM(centre) / 2.

    Returns:
        Smoothed array; NaNs are ignored, +-inf propagate.
    """
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
    """Running mean of a binned curve over smoothing_factor x envelope FWHM.

    Args:
        centres: bin centres [muHz], ascending.
        values: curve at the bin centres.
        smoothing_factors: scalar or list of window widths in envelope FWHM.

    Returns:
        {smoothing_factor: smoothed array}.
    """
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
