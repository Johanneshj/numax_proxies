import numpy as np
from numpy.typing import NDArray
import time as t
import itertools
from typing import Union, List, Tuple
from ...data_preparation.dataclasses import COVConfig
import matplotlib.pyplot as plt

def log_numax_binning(
        frequency : NDArray, 
        power : NDArray, 
        min_freq : float = 100,
        max_freq : float = None,
        overlap_scales : Union[float, List[float]] = 6, 
        width_factors : Union[float, List[float]] = 1,
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

def calculate_CoVs(
        bin_centers : List[NDArray], 
        power_windows : List[NDArray]
):
    """Calculate Coefficients of Variation for each bin center"""

    CoVs_full = [
        np.array([calculate_CoV(pw) for pw in pws])
        for pws in power_windows
    ]

    return CoVs_full

def calculate_CoV(
        power_window : NDArray
):
    """
    Calculate CoV per bin as standard deviation / mean.
    Bin defined by center and width with closed left ends and open right ends.

    Input:
        center :: center of frequency bin
        width :: width of frequency bin
        frequency :: list of frequencies in muHz
        power :: PSD

    Return:
        CoV :: Coefficient of Variation in bin
    """
   
    # -----------------------
    # Fail safes
    # -----------------------

    # Is bin length zero?
    if len(power_window) == 0:
        CoV = 1.
        return CoV

    # Is bin power atleast length 1?
    if len(power_window) <= 1:
        CoV = 1.
        return CoV

    # Compute mean
    mean = np.mean(power_window)

    # Is mean finite and non-zero?
    if not np.isfinite(mean) or mean == 0:
        CoV = 1.
        return CoV

    # Compute SD
    std = np.std(power_window, ddof=1)

    # is SD finite?
    if not np.isfinite(std):
        CoV = 1.
        return CoV

    # Then we calculate CoV
    CoV = std / mean

    return CoV

def smoothing_func(
    bin_centers: NDArray,
    CoVs: NDArray,
    smoothing_factors: Union[float, list[float]] = 5.0,
) -> list[NDArray]:
    """Smooth CACF values with FWHM of potential oscillation envelope"""

    bin_centers = np.asarray(bin_centers)
    CoVs = np.asarray(CoVs)

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
        smoothed = np.empty_like(CoVs)

        for i, (center, width) in enumerate(zip(bin_centers, bin_widths)):

            lower = center - width / factor
            upper = center + width / factor

            mask = (bin_centers >= lower) & (bin_centers <= upper)

            smoothed[i] = np.nanmean(CoVs[mask]) if np.any(mask) else np.nan

        all_smoothed.append(smoothed)
        
    return all_smoothed

def smooth_CoVs(
       bin_centers  :   list[NDArray],
       CoVs         :   list[NDArray],
       cov_config   :   COVConfig
):
    """Compute smoothed CoV values"""
    smoothed_CoVs = [
        smoothing_func(centers, CoV_vals, cov_config.smoothing_factor)
        for centers, CoV_vals in zip(bin_centers, CoVs)
    ]

    return smoothed_CoVs