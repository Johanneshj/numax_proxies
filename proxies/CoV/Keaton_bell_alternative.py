# Evaluate coefficients of variation according to Bell et al. (2019)
# With FAP for CoV significance.

import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def evaluate_faps(n_bins):
    """
    Evaluates the False Alarm Probability (FAP) at 0.1% level based on the simulation
    results published by Bell et al. (2019)
    
    Input:
        n_bins :: the number of bins corresponding to the evaluation of a given CoV value
        for which the FAP is required.
    """
    fap_array = np.array([1.830,1.942,1.865,1.664,1.486,1.341,1.227,1.157,1.108,1.0]) 
    n_bins_array = np.array([4,8,16,32,64,128,256,512,1024,8196])
    fap = interp1d(n_bins_array,fap_array,fill_value='extrapolate',bounds_error=False)(n_bins)

    exceed_tmp = np.where(n_bins > 1024)[0]
    if len(exceed_tmp) > 0:
        fap[exceed_tmp] = interp1d(n_bins_array,fap_array,'linear',fill_value='extrapolate',bounds_error=False)(n_bins[exceed_tmp])

    return fap

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
        bin_size = 0
        return [CoV, bin_size]

    bin_power = power[index_bin]
    bin_power = bin_power[np.isfinite(bin_power)]

    # Is bin power atleast length 1?
    if len(bin_power) <= 1:
        CoV = 1.
        bin_size = 0
        return [CoV, bin_size]

    mean = np.mean(bin_power)

    # Is mean finite and non-zero?
    if not np.isfinite(mean) or mean == 0:
        CoV = 1.
        bin_size = 0
        return [CoV, bin_size]

    std = np.std(bin_power, ddof=1)

    # is SDV finite?
    if not np.isfinite(std):
        CoV = 1.
        bin_size = 0
        return [CoV, bin_size]

    # Then we calculate CoV
    bin_size = len(index_bin)
    CoV = std / mean
    return [CoV, bin_size]

def bin_spectrum(frequency=None, power=None, overlap_factor=6, min_freq=1.0):
    """
    Binning of spectrum based on formalism by Viani et al. (2018).
    Spectrum is binned in segments with size 0.267 * numax^0.764 (Yu et al. 2018), where numax is the central frequency of the bin.
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

    # Create the bin centers using a width proportional to numax, with numax assumed as the
    # center of the bin
    bin_centers = [min_freq]  # starting point in muHz (1 muHz in Viani)
    bin_widths = [0.267 * bin_centers[0] ** 0.764]
    while bin_centers[-1] < frequency[-1]:
        next_center = bin_centers[-1] + (0.267 * bin_centers[-1] ** 0.764) / overlap_factor
        bin_centers.append(next_center)
        width = 0.267 * next_center**0.764
        bin_widths.append(width)

    # Use the bin centers and widths to bin the spectrum
    CoVs = []
    bin_sizes = []
    for center, width in zip(bin_centers, bin_widths):
        CoV = calculate_CoV(center, width, frequency, power)
        CoVs.append(CoV[0])
        bin_sizes.append(CoV[1])

    # Regularize the data
    CoVs = np.asarray(CoVs)
    bin_sizes = np.asarray(bin_sizes)
    good_indices = np.isfinite(CoVs)
    CoVs = CoVs[good_indices]
    bin_sizes = bin_sizes[good_indices]
    bin_centers = np.asarray(bin_centers)
    bin_centers = bin_centers[good_indices]
    faps_CoV = evaluate_faps(bin_sizes)

    # Return bin_centers (frequencies) and associated CoV values and bin sizes
    return bin_centers, CoVs, faps_CoV


def smooth_CoV_values(bin_centers, CoVs):
    """
        Smooth CoV values following Viani et al. (2018).
        Smoothing window has size 0.66*bin_central^0.88

        Inputs:
            bin_centers     : central frequency of bins
            CoVs            : raw CoV values

        Outputs:
            smoothed_CoVs   : smoothed CoV values
    """
    smoothed_CoVs = [
        smoothing_func(center, bin_centers, CoVs, 1) for center in bin_centers
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

def numax_estimate_CoV(bin_centers, smoothed_CoVs, CoVs, faps_CoV, initial_numax=None):
    """
    Here we estimate numax. 
    First we estimate location of numax.
    Location estimated either from fitting Gaussian (requires numax init).
    Otherwise location of maximum CoV is taken as numax.
    Then uncertainty on numax is estimated as std of bins that make up the oscillation envelope.

    Inputs:
        bin_centers     : list of bin centers
        smoothed_CoVs   : list of smoothed CoV values
        CoVs            : list of unsmoothed (original) CoV values
        faps_CoV        : the false alarm probability associated to each CoV value
        initial_numax   : initial guess on numax in muHz
        
    Output:
        CoV_numax       : numax estimate
        CoV_numax_err   : error on numax
        popt            : fit values from Gaussian fit (None if initial_numax=None)
        successful_fit   : flag if Gaussian fit was successful (False if initial_numax=None)
    """

  
    # Do Gaussian fit if initial_numax is given. In case it is not provided, evaluate it from the
    # maximum of the distribution, by excluding signal below the FAP level and above the detection limit
    # threshold for solar-like oscillations (Bell et al. 2019).
    if initial_numax:
        numax_init = initial_numax
    else:
        difference = CoVs - faps_CoV
        good_indices = (difference > 0.0)

        if len(CoVs[good_indices]) > 0:
            bin_centers_local = bin_centers[good_indices]
            smoothed_CoVs_local = smoothed_CoVs[good_indices]
            CoVs_local = CoVs[good_indices]
            upper_CoV_limit = 2.69*bin_centers_local**0.154 
            good_indices2 = (CoVs_local < upper_CoV_limit)
            
            # Exclude potential spikes from CoV by adopting an empirical upper limit
            # for solar-like oscillations
            if len(CoVs_local[good_indices2]) < len(bin_centers_local) and len(CoVs_local[good_indices2]) > 0:
                bin_centers_clean = bin_centers_local[good_indices2]
                smoothed_CoVs_clean = smoothed_CoVs_local[good_indices2]
                numax_init = bin_centers_clean[np.argmax(smoothed_CoVs_clean)]
            else:
                numax_init = bin_centers_local[np.argmax(smoothed_CoVs_local)]
        else:
            print('CoV at numax estimate below FAP level.')
            return np.nan, np.nan, [], False

    
    # For the Gaussian fitting feed in the smoothed CoV cleaned for possible spikes
    # and use only the region of interest
    upper_CoV_limit = 2.69*bin_centers**0.154 
    good_indices = (CoVs < upper_CoV_limit)
    bin_centers_clean = bin_centers[good_indices]
    smoothed_CoVs_clean = smoothed_CoVs[good_indices]
    
    width = 0.66 * numax_init**0.88
    lower = numax_init - width
    upper = numax_init + width
    indices = np.where((bin_centers_clean >= lower) & (bin_centers_clean <= upper))[0]
    numax_init, popt, successful_fit = fit_for_numax(bin_centers_clean[indices],
                                                    smoothed_CoVs_clean[indices],
                                                    numax_init)
    
    # Determine which bins contribute to numax envelope
    if successful_fit:
        CoV_numax = popt[2]
        CoV_numax_error = popt[1]
    else:
        width = 0.66 * numax_init**0.88
        lower = numax_init - width / 2
        upper = numax_init + width / 2
        indices = np.where((bin_centers_clean >= lower) & (bin_centers_clean <= upper))[0]

        numerator = np.nansum(bin_centers_clean[indices] * smoothed_CoVs_clean[indices])
        denominator = np.nansum(smoothed_CoVs_clean[indices])
        # print(numerator, denominator)
        # Double check in case CoV fails
        if denominator == 0 or np.isnan(denominator) or denominator is None:
            CoV_numax = -1
            CoV_numax_error = -1
        else:
            CoV_numax = numerator / denominator
            CoV_numax_error = np.abs(np.nanstd(bin_centers_clean[indices], ddof=1))
    
    return CoV_numax, CoV_numax_error, popt, successful_fit

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
            successful_fit  : flag indicating if fit was successful
    """

    successful_fit = False # Flag to check if fit was good
    # We do a "try" here in case fits fails or is underresolved
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)

        # Safety check
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        # Initial guesses
        amp0 = np.max(y)
        w0 = 0.66*numax_init**0.88 #0.1
        p0s = [amp0, w0, numax_init]

        # Fit
        popt, pcov = curve_fit(
            gaussian_with_offset,
            x,
            y,
            p0=p0s,
        )
        numax = popt[1]
        successful_fit = True

        return numax, popt, successful_fit

    except (RuntimeError, OptimizeWarning, ValueError) as e:
        # If fit fails return numax as location of highest peak in a region around numax_init
        print(f'Gaussian fit for CoV method failed because of {e}, falling back to location of maximum CoV.')
        mask = (x >= numax_init - window) & (x <= numax_init + window)
        return x[mask][np.argmax(y[mask])], np.nan, successful_fit

def gaussian_with_offset(x, A, sigma, mu):
    """Gaussian function"""
    return 1 + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

# def gaussian_func(x, A, mu, sigma, c, d, e):
#     """
#         Gaussian function with an added linear term to account for background

#         Inputs:
#             x   : x values (frequencies)
#             A   : amplitude of Gaussian
#             mu  : mean of Gaussian (numax)
#             sigma: standard deviation of Gaussian
#             c, d, e : linear coefficients for background

#         Output:
#             y   : y values (CoV)
#     """
#     return A * np.exp(-(x - mu) ** 2 / (2 * sigma**2)) + c * x**2 + d * x + e