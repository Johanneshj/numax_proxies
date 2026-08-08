from scipy.ndimage import median_filter
import numpy as np
from numpy.typing import NDArray
import time as t
import itertools
from typing import Union, List, Tuple
from ...data_preparation.dataclasses import ACFConfig
from scipy.signal import correlate
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

def calculate_relative_power(frequency : NDArray, power : NDArray):
    """Subtract and normalize PSD by median filter (Viani+ 2019)"""

    if np.max(frequency) > 300:
        ws = 100  # muHz
    else:
        ws = 10  # muHz

    df = np.median(np.diff(frequency))
    wp = int(ws / df)

    med_filter = median_filter(power, size=wp, mode="reflect")
    rel_power = (power - med_filter) / med_filter

    return rel_power, med_filter

def calculate_two_dim_ACF(frequency : NDArray, power : NDArray, acf_config : ACFConfig):
    """
    Calculate 2D autocorrelation function:
        
    Use a sliding window where each window slides a distance equal to some fraction of a 
    potential delta_nu at the center of the previous bin.
    Each bin is sized as if a potential delta_nu could fit in it.

    Input:
        frequency :: list of frequencies in muHz
        power :: power normalized to relative power

    Output:
        acf :: 2D AutoCorrelation Function
        freq_windows :: binned frequency list
            -> needed for plotting and collapsed ACF.
    """
    
    start = t.time()
  
    bin_centers, freq_windows, power_windows = log_numax_binning(
        frequency           =   frequency,
        power               =   power,
        overlap_scales      =   acf_config.overlap_scale,
        min_freq            =   acf_config.min_freq,
        max_freq            =   acf_config.max_freq,
        width_factors       =   acf_config.width_factor,
        smoothing_factors   =   acf_config.smoothing_factor
    )

    # Calculate acf for each segment
    acfs = [abs_acf(pw) for pw in power_windows]

    end = t.time()

    print(f'2D ACF calculation time: {np.round(end-start, 3)} seconds')

    return bin_centers, freq_windows, acfs

def log_numax_binning(
        frequency : NDArray, 
        power : NDArray, 
        overlap_scales : Union[float, List[float]] = 6, 
        min_freq : float = 100, 
        max_freq : float = None, 
        width_factors : Union[float, List[float]] = 1,
        smoothing_factors : Union[float, List[float]] = 1
) -> tuple[
        list[np.ndarray],
        list[list[np.ndarray]],
        list[list[np.ndarray]],
    ]:   
    """
        Perform sliding window as used for CoV method in Viani+ 2019.
        Basically, around each bin center the bin size will be defined as
        the width a potential delta_nu.

        Inputs:
            frequency       : frequency array
            power           : power array
            overlap_scale   : control how much overlap there is between bins
                                and therebby the number of bins.
            min_num_points  : minimum number of points in a bin.
            min_freq        : minimum frequency (microHz) for first bin center.

        The function "binning_parameters" returns overlap_scale, min_num_points, min_freq.
    """
    # Standardize inputs to lists
    scales = [float(overlap_scales)] if isinstance(overlap_scales, (int, float)) else [float(s) for s in overlap_scales]
    factors = [float(width_factors)] if isinstance(width_factors, (int, float)) else [float(w) for w in width_factors]

    # Smoothing factors will only be used here to determine the dimensionality of the res_list
    smoothing_factors = ([float(smoothing_factors)] if isinstance(smoothing_factors, (int, float)) else [float(s) for s in smoothing_factors])

    if max_freq is None:
        max_freq = frequency[-1]

    bcs:    List[NDArray] = []
    fs:     List[NDArray] = []
    ps:     List[NDArray] = []

    # Loop of overlap scales and width factors
    for scale, factor in itertools.product(scales, factors):
        init_center = float(max(frequency[0], min_freq))
        bin_centers = [init_center]
        init_width = float(factor * 0.267 * (init_center ** 0.764))
        bin_widths = [init_width]

        # While loop generating bins
        while bin_centers[-1] < max_freq:
            step_size = (factor * 0.267 * (bin_centers[-1] ** 0.764)) / scale
            next_center = bin_centers[-1] + step_size

            if next_center > max_freq:
                break

            bin_centers.append(next_center)
            next_width = factor * 0.267 * next_center**0.764
            bin_widths.append(next_width)

        # Calculate mean power within each bin
        freqs = []
        powers = []

        # Calculate power in bin
        for center, width in zip(bin_centers, bin_widths):
            half_w = width / 2.0
            idx_start = np.searchsorted(frequency, center - half_w, side="left")
            idx_end = np.searchsorted(frequency, center + half_w, side="right")
            
            freqs.append(frequency[idx_start:idx_end])
            powers.append(power[idx_start:idx_end])

        bcs.append(bin_centers)
        fs.append(freqs)
        ps.append(powers)

    return bcs, fs, ps

def abs_acf(power_windows: list[np.ndarray]) -> list[np.ndarray]:
    """
    Autocorrelation by means of scipy correlate.
    Normalization tailored to log sliding windows.

    Input:
        power_windows :: PSD values

    Output:
        np.abs(corr) * scaling :: normalized absolute autocorrelation
    """
    def _compute_single_acf(x: NDArray) -> NDArray:
        if len(x) == 0:
            return np.array([], dtype=float)

        # Center data
        x_centered = x - np.mean(x)

        # Can we do more here to minimize spectral leakage
        # x_centered *= hann(len(x_centered), sym=True)

        # Perform autocorrelation by means of FFT
        corr = correlate(x_centered, x_centered, mode='full', method='fft')

        # Grab positive lags only
        corr = corr[corr.size // 2 :] 

        # Scale with 1/sqrt(N)
        scaling = 1 / np.sqrt(len(x))

        # Return abs(corr) times scaling
        return np.abs(corr) * scaling

    return [_compute_single_acf(w) for w in power_windows]

def collapse_segment(seg : NDArray) -> float:
    """Collapse segment of total ACF and return mean."""
    # Ignore first index which always has ACF = 1
    seg = seg[1:]

    # Check length of segment
    if len(seg) < 1:
        return np.nan
    
    # Check std
    std = np.nanstd(seg, ddof=1)
    if std == 0:
        return np.nan
    
    # Return collapsed acf
    mean = np.nanmean(seg)
    return mean

def smoothing_func(
    bin_centers: NDArray,
    collapsed_acf: NDArray,
    smoothing_factors: Union[float, list[float]] = 5.0,
) -> list[NDArray]:
    """Smooth CACF values with FWHM of potential oscillation envelope"""

    bin_centers = np.asarray(bin_centers)
    collapsed_acf = np.asarray(collapsed_acf)

    # Check how many smoothing factors we want to investigate
    factors = (
        [float(smoothing_factors)]
        if isinstance(smoothing_factors, (int, float))
        else [float(s) for s in smoothing_factors]
    )

    # Empty list
    all_smoothed = []

    # Smooth
    bin_widths = 0.66 * bin_centers**0.88
    for factor in factors:
        smoothed = np.empty_like(collapsed_acf)

        for i, (center, width) in enumerate(zip(bin_centers, bin_widths)):

            lower = center - width / factor
            upper = center + width / factor

            mask = (bin_centers >= lower) & (bin_centers <= upper)

            smoothed[i] = np.nanmean(collapsed_acf[mask]) if np.any(mask) else np.nan

        all_smoothed.append(smoothed)
        
    return all_smoothed

def collapsed_acf(
    bin_centers:    list[NDArray],
    acfs:           list[NDArray],
    acf_config:     ACFConfig
) -> tuple[
    list[np.ndarray],
    list[list[np.ndarray]],
]:
    all_cacfs = [
        np.array([collapse_segment(seg) for seg in acf])
        for acf in acfs
    ]

    smoothed_cacfs = [
        smoothing_func(centers, cacf, acf_config.smoothing_factor)
        for centers, cacf in zip(bin_centers, all_cacfs)
    ]

    flat_bin_centers = []
    norm_unsmoothed = []
    norm_smoothed = []

    for bcs, u_cacf, scacfs in zip(bin_centers, all_cacfs, smoothed_cacfs):
        u_norm = normalize_0_1(u_cacf)
        
        for scacf in scacfs:
            flat_bin_centers.append(bcs)
            norm_unsmoothed.append(u_norm)
            norm_smoothed.append(normalize_0_1(scacf))

    return flat_bin_centers, norm_unsmoothed, norm_smoothed

def normalize_0_1(x : NDArray):
    """Normalize between 0 and 1."""
    x = np.asarray(x)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    return (x - xmin) / (xmax - xmin)