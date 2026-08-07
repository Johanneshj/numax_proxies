# Functions for binning spectrum and calculating CoV values
# Also smoothing the CoV values
# Finally also calcuing weighted mean for numax estimate (Viani et al. 2018)

import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from typing import Optional
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray


def calculate_CoV(center, width, frequency, power):
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
    lower = center - width / 2
    upper = center + width / 2
    index_bin = np.where((frequency >= lower) & (frequency < upper))[0]

    # -----------------------
    # Fail safes
    # -----------------------

    # Is bin length zero?
    if len(index_bin) == 0:
        CoV = 1.
        return CoV

    bin_power = power[index_bin]
    bin_power = bin_power[np.isfinite(bin_power)]

    # Is bin power atleast length 1?
    if len(bin_power) <= 1:
        CoV = 1.
        return CoV
    
    mean = np.mean(bin_power)

    # Is mean finite and non-zero?
    if not np.isfinite(mean) or mean == 0:
        return np.nan

    std = np.std(bin_power, ddof=1)

    # is SDV finite?
    if not np.isfinite(std):
        return np.nan

    # Then we calculate CoV
    CoV = (std / mean)

    return CoV




def bin_spectrum(frequency=None, power=None, min_freq : Optional[float] = None, 
                 overlap_factor : Optional[float] = None, use_linear_bins : Optional[bool] = False,
    ):
    """
    Binning of spectrum based on formalism by Viani et al. (2018).
    Spectrum is binned in segments with size 0.267 * numax^0.764 (Yu et al. 2018),
    where numax is the central frequency of the bin.
    We start from 1 muHz and then move the window 1/6 of the previous window size.

    Input:
        frequency :: list of frequencies in muHz
        power :: PSD
        overlap_factor :: factor for sliding window
            Viani et al. (2018) had overlap_factor=6,
            but seems that higher values can improve without too much computational cost.

    Return:
        binned_frequency
        binned_power
        mean_power
    """

    if use_linear_bins:
        freq_windows, power_windows = linear_binning(
            frequency=frequency,
            power=power
        )
        stds = np.nanstd(power_windows, axis=1, ddof=1)
        means = np.nanmean(power_windows, axis=1)
        CoVs = stds / means
        bin_centers = np.nanmean(freq_windows, axis=1)
        return bin_centers, CoVs


    if min_freq is None:
        min_freq = 1.0

    if overlap_factor is None:
        overlap_factor = 6.0

    # In this while-loop we'll create the bin centers
    bin_centers = [min_freq]  # starting point is 0.1 muHz (1 muHz in Viani)
    bin_widths = [0.267 * bin_centers[0] ** 0.764]
    while bin_centers[-1] < frequency[-1]:
        next_center = bin_centers[-1] + (0.267 * bin_centers[-1] ** 0.764) / overlap_factor
        bin_centers.append(next_center)
        width = 0.267 * next_center**0.764
        bin_widths.append(width)

    # Well use the bin centers and widths to bin the spectrum
    CoVs = [
        calculate_CoV(center, width, frequency, power)
        for center, width in zip(bin_centers, bin_widths)
    ]

    # Safe data
    bin_centers = np.asarray(bin_centers, dtype=float)
    CoVs = np.asarray(CoVs, dtype=float)
    valid = np.isfinite(bin_centers) & np.isfinite(CoVs)
    bin_centers = bin_centers[valid]
    CoVs = CoVs[valid]

    # Return bin_centers (frequencyes) and associated CoV values
    return np.asarray(bin_centers), np.asarray(CoVs)


def smooth_CoV_values(bin_centers, CoVs, smoothing_width_factor : Optional[float]=None):
    """
        Smooth CoV values following Viani et al. (2018).
        Smoothing window has size 0.66*bin_central^0.88

        Inputs:
            bin_centers     : central frequency of bins
            CoVs            : raw CoV values

        Outputs:
            smoothed_CoVs   : smoothed CoV values
    """
    if smoothing_width_factor is None:
        smoothing_width_factor = 1.0

    smoothed_CoVs = [
        smoothing_func(center, bin_centers, CoVs, smoothing_width_factor) for center in bin_centers
    ]

    # Return smoothed CoVs
    return np.asarray(smoothed_CoVs)


def smoothing_func(center, bin_centers, CoVs, smoothing_width_factor):
    '''
        Smoothing function as described in Viani et al. (2018).
        Spectrum is smoothed as medium of bins with 
        size equal to width of potential numax envelope.

        Inputs:
            center          : center of bin in muHz
            bin_centers     : list of our bin centers
            CoVs            : CoV values
        
        Output:
            smoothed_val    : smoothed val in bin
    '''

    width = smoothing_width_factor * 0.66 * center**0.88
    lower = center - width / 2
    upper = center + width / 2
    indices = np.where((bin_centers >= lower) & (bin_centers < upper))[0]
    if len(indices) > 0 and np.sum(~np.isnan(CoVs[indices])) > 0:
        smoothed_val = np.nanmean(CoVs[indices])
    else:
        smoothed_val = np.nan

    return smoothed_val


def numax_estimate_CoV(bin_centers, smoothed_CoVs, initial_numax=None):
    """
        Here we estimate numax. 
        First we estimate location of numax.
        Location estimated either from fitting Gaussian (requires numax init).
        Otherwise location of maximum CoV is taken as numax.
        Then uncertainty on numax is estimated as std of bins that make up the oscillation envelope.

        Inputs:
            bin_centers     : list of bin centers
            smoothed_CoVs   : list of smoothed CoV values
            initial_numax   : initial guess on numax in muHz
        
        Output:
            CoV_numax       : numax estimate
            CoV_numax_err   : error on numax
            popt            : fit values from Gaussian fit (None if initial_numax=None)
            succesful_fit   : flag if Gaussian fit was succesful (False if initial_numax=None)
    """
    # Masking array
    # mask = (bin_centers > 0.1) # can be 1 muHz
    bin_centers = bin_centers
    smoothed_CoVs = smoothed_CoVs
    
    # Do Gaussian fit in initial_numax given
    if initial_numax:
        numax_init = bin_centers[
            np.argmin(
                np.abs(
                    bin_centers - initial_numax
                )
            )
        ]
        numax_init, popt, succesful_fit = fit_for_numax(
            bin_centers,
            smoothed_CoVs,
            numax_init
        )
        numax_init = numax_init # 10 ** numax_init
    else:
        numax_init = bin_centers[np.argmax(smoothed_CoVs)]
        numax_init, popt, succesful_fit = fit_for_numax(
            bin_centers,
            smoothed_CoVs,
            numax_init
        )
        numax_init = numax_init # 10 ** numax_init
    
    # Determine which bins contribute to numax envelope
    width = 0.66 * numax_init**0.88
    lower = numax_init - width / 2
    upper = numax_init + width / 2
    indices = np.where((bin_centers >= lower) & (bin_centers <= upper))[0]

    numerator = np.nansum(bin_centers[indices] * smoothed_CoVs[indices])
    denominator = np.nansum(smoothed_CoVs[indices])

    # Double check in case CoV fails
    if denominator == 0 or np.isnan(denominator) or denominator is None:
        CoV_numax = np.nan
        CoV_numax_error = np.nan
    else:
        CoV_numax = numerator / denominator
        CoV_numax_error = np.nanstd(bin_centers[indices], ddof=1)
    return CoV_numax, CoV_numax_error, popt, succesful_fit

def fit_for_numax(x, y, numax_init, window=500):
    """
        Fit Gaussian to CoV spectrum if initial guess is specified

        Inputs:
            x           : x values (frequencies)
            y           : y values (CoV)
            numax_init  : initial numax guess
            window      : in case fits fails we take numax as most significant 
                            peak in window around numax_init
        
        Outputs:
            numax           : numax estimate from fitting
            popt            : fit values
            succesful_fit   : flag indicating if fit was succesful
    """
    succesful_fit = False # Flag to check if fit was good
    # We do a "try" here in case fits fails or is underresolved
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)

        # Safety check
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        # Initial guesses
        amp0 = 0.9 * np.max(y)
        numax0 = numax_init
        w0 = (2/3) * numax0 ** (22/25)
        p0s = [amp0, w0, numax0]

        # Bounds
        lower_bounds = [0, 0, np.min(x)]
        upper_bounds = [np.max(y), 2*w0, np.max(x)]
        
        # Fit
        popt, pcov = curve_fit(
            gaussian_with_offset,
            x,
            y,
            p0=p0s,
            bounds=(
                lower_bounds,
                upper_bounds
            )
        )
        numax = popt[2]
        succesful_fit = True
        return numax, popt, succesful_fit

    except (RuntimeError, OptimizeWarning, ValueError) as e:
        print(f'fit failed because of "{e}"')
        # If fit fails return numax as location of highest peak in a region around numax_init
        mask = (x >= numax_init - window) & (x <= numax_init + window)
        return x[mask][np.argmax(y[mask])], np.nan, succesful_fit

def normalize_0_1(x):
    x = np.asarray(x)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    return (x - xmin) / (xmax - xmin)

def gaussian_with_offset(x, A, sigma, mu):
    """Gaussian function"""
    return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def linear_binning(frequency : NDArray, power : NDArray):
    """Linear sliding window (Viani+ 2019)"""
    df = np.mean(np.diff(frequency))

    if np.max(frequency) > 300:
        window_size_muHz = 10  # window width
        overlap_muHz = 9  # overlap
    else:
        window_size_muHz = 50  # window width
        overlap_muHz = 40  # overlap

    window_size = int(window_size_muHz / df)
    step = int((window_size_muHz - overlap_muHz) / df)
    if step <= 0:
        step = 10  # failsafe
    freq_windows = sliding_window_view(frequency, window_shape=window_size)[::step]
    power_windows = sliding_window_view(power, window_shape=window_size)[::step].copy()
    return freq_windows, power_windows