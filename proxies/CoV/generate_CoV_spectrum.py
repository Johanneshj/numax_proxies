# Python functions for computing CoV spectra for PSDs of solar-like oscillators.
# Method is fully vectorized, meaning, different overlap_scales, width_factors, and smoothing factors can be explored
# without extensize computational overhead.
# Final output is returned by "smooth_CoVs"; dictionary with entries (overlap, width, smoothing_factor),
# containing bin_centers, raw_CoVs, and smoothed CoVs.

import numpy as np
from numpy.typing import NDArray
import time as t
import itertools
from typing import Dict, List, Tuple, Union
from ...data_preparation.dataclasses import COVConfig

ParamKey = Tuple[float, float]  # (overlap_scale, width_factor)
BinningResult = Dict[str, List[NDArray]]
FullParamKey = Tuple[float, float, float] # (overlap_scale, width_factor, smoothing_factor)

def log_numax_binning(
        frequency : NDArray, 
        power : NDArray, 
        min_freq : float = 100, 
        max_freq : float = None, 
        overlap_scales : Union[float, List[float]] = 6, 
        width_factors : Union[float, List[float]] = 1
) -> Dict[ParamKey, BinningResult]: 

    """
        Perform sliding window as used for CoV method in Viani+ 2019.
        Basically, around each bin center the bin size will be defined as
        the width a potential delta_nu.

        Parameters:
            frequency       :   frequency array
            power           :   power array
            min_num_points  :   minimum number of points in a bin.
            min_freq        :   minimum frequency (microHz) for first bin center.
            overlap_scales  :   control how much overlap there is between bins.
            width_factors   :   control width of bins
        
        Returns:
            binning_results :   Dictionary with entries corresponding to each combination of 
                                overlap_scale and width_factor.
                                Each entry has bin_centers and power within bins.   
    """
    # Standardize inputs to lists
    scales = [float(overlap_scales)] if isinstance(overlap_scales, (int, float)) else [float(s) for s in overlap_scales]
    factors = [float(width_factors)] if isinstance(width_factors, (int, float)) else [float(w) for w in width_factors]

    if max_freq is None:
        max_freq = frequency[-1]

    binning_results = {}

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

        binning_results[(scale, factor)] = {
            "bin_centers": np.array(bin_centers),
            "freq_windows": freqs,
            "power_windows": powers
        }

    return binning_results

def calculate_CoVs(
    binning_results: Dict[ParamKey, dict],
    inplace: bool = False
) -> Dict[ParamKey, dict]:
    """
    Calculate Coefficient of Variation (CoV) for each frequency bin window across 
    all parameter combinations in binning_results.

    Parameters:
    -----------
    binning_results : Dict[ParamKey, dict]
        Dictionary keyed by (overlap_scale, width_factor) containing - obtained from "log_numax_binning":
        - "bin_centers": 1D array of bin center frequencies
        - "freq_windows": List of 1D frequency arrays per bin
        - "power_windows": List of 1D power arrays per bin
    inplace : bool
        If True, updates input dictionary directly. If False, returns a new dict.

    Returns:
    --------
    Dict[ParamKey, dict]
        Dictionary where each entry contains:
        - "bin_centers": 1D array of frequency bin centers
        - "power_windows": Original power windows
        - "cov": 1D array of CoV values corresponding to each bin center
    """
    results = binning_results if inplace else {k: v.copy() for k, v in binning_results.items()}

    for param_key, data in results.items():
        power_windows = data["power_windows"]
        
        # Calculate CoV scalar for each power window
        cov_values = np.array([calculate_CoV(pw) for pw in power_windows])

        # Store CoV array in the dictionary
        data["cov"] = cov_values

    return results

def calculate_CoV(
        power_window : NDArray
):
    """
    Calculate CoV per bin as standard deviation / mean.
    Bin defined by center and width with closed left ends and open right ends.

    Parameters:
        center : center of frequency bin
        width : width of frequency bin
        frequency : list of frequencies in muHz
        power : PSD

    Return:
        CoV : Coefficient of Variation in bin
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
    smoothing_factors: Union[float, List[float]] = 5.0,
) -> Dict[float, NDArray]:
    """
    Smooth CoV values with FWHM of potential oscillation envelope.
    
    Parameters:
        bin_centers         :   1D array with centers of bins
        CoVs                :   1D array of CoV values
        smoothing_factors   :   list of smoothing factors; multiplicative factor to FWHM.

    Returns:
        Dict mapping each smoothing_factor float -> 1D smoothed CoV array.
    """
    bin_centers = np.asarray(bin_centers)
    CoVs = np.asarray(CoVs)

    factors = (
        [float(smoothing_factors)]
        if isinstance(smoothing_factors, (int, float))
        else [float(s) for s in smoothing_factors]
    )

    bin_widths = 0.66 * (bin_centers ** 0.88)
    smoothed_results = {}

    for factor in factors:
        smoothed = np.empty_like(CoVs)
        
        # Total window width = factor * bin_widths -> half_width = (factor * bin_widths) / 2.0
        half_widths = (factor * bin_widths) / 2.0

        lowers = bin_centers - half_widths
        uppers = bin_centers + half_widths

        # Binary search window bounds
        idx_starts = np.searchsorted(bin_centers, lowers, side="left")
        idx_ends = np.searchsorted(bin_centers, uppers, side="right")

        for i, (start, end) in enumerate(zip(idx_starts, idx_ends)):
            if start < end:
                smoothed[i] = np.nanmean(CoVs[start:end])
            else:
                smoothed[i] = np.nan

        smoothed_results[factor] = smoothed

    return smoothed_results

def smooth_CoVs(
    cov_results: Dict[Tuple[float, float], dict],
    cov_config: COVConfig
) -> Dict[FullParamKey, dict]:
    """
    Applies frequency-dependent smoothing to CoV curves across all smoothing factors.

    Parameters:
    -----------
    cov_results : Dict[Tuple[float, float], dict]
        Dictionary keyed by (overlap_scale, width_factor) containing 'bin_centers' and 'cov'.
    cov_config : COVConfig
        Configuration containing smoothing_factor(s).

    Returns:
    --------
    Dict[FullParamKey, dict]
        Dictionary keyed by (overlap_scale, width_factor, smoothing_factor) containing:
        - "bin_centers": 1D array of frequency bin centers
        - "cov": Raw 1D CoV array
        - "cov_smoothed": Smoothed 1D CoV array
        - "cov_norm_smoothed": Min-max normalized smoothed CoV array [0, 1]
    """
    smoothing_factors = cov_config.smoothing_factor
    results = {}

    for (scale, width_factor), data in cov_results.items():
        bin_centers = data["bin_centers"]
        cov_raw = data["cov"]

        # Compute smoothed CoVs for all requested smoothing factors
        smoothed_by_factor = smoothing_func(
            bin_centers=bin_centers,
            CoVs=cov_raw,
            smoothing_factors=smoothing_factors
        )

        # Store entries indexed by the full 3-parameter key
        for smooth_factor, cov_smoothed in smoothed_by_factor.items():
            full_key = (scale, width_factor, smooth_factor)
            results[full_key] = {
                "bin_centers": bin_centers,
                "cov": cov_raw,
                "cov_smoothed": cov_smoothed
            }

    return results