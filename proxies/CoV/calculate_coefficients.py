# Functions for binning spectrum and calculating CoV values
# Also smoothing the CoV values
# Finally also calcuing weighted mean for numax estimate (Viani et al. 2018)

import numpy as np

def calculate_CoV(center, width, frequency, power):
    '''
        Calculate CoV per bin as standard deviation / mean.
        Bin defined by center and width with closed left ends and open right ends.

        Input:
            center :: center of frequency bin
            width :: width of frequency bin
            frequency :: list of frequencies in muHz
            power :: PSD

        Return:
            CoV :: Coefficient of Variation in bin
    '''
    lower = center - width / 2
    upper = center + width / 2
    index_bin = np.where((frequency >= lower) & (frequency < upper))[0]
    if len(index_bin) > 0:
        bin_power = power[index_bin]
        mean = np.mean(bin_power)
        std = np.std(bin_power, ddof=1)
        CoV = std / mean
    else:
        CoV = np.nan
    return CoV


def bin_spectrum(frequency=None, power=None):
    '''
        Binning of spectrum based on formalism by Viani et al. (2018).
        Spectrum is binned in segments with size 0.267 * numax^0.764 (Yu et al. 2018),
        where numax is the central frequency of the bin.
        We start from 1 muHz and then move the window 1/6 of the previous window size.

        Input:
            frequency :: list of frequencies in muHz
            power :: PSD

        Return:
            binned_frequency
            binned_power
            mean_power
    '''

    # In this while-loop we'll create the bin centers
    bin_centers = [1] # starting point is 1 muHz
    bin_widths = [0.267 * bin_centers[0]**0.764]
    while bin_centers[-1] < frequency[-1]:
        next_center = bin_centers[-1] + (0.267 * bin_centers[-1]**0.764) / 6
        bin_centers.append(next_center)
        width = 0.267 * next_center**0.764
        bin_widths.append(width)

    # Well use the bin centers and widths to bin the spectrum
    CoVs = [calculate_CoV(center, width, frequency, power) for center, width in zip(bin_centers, bin_widths)]

    return np.asarray(bin_centers), np.asarray(CoVs)

def smooth_CoV_values(bin_centers, CoVs):
    '''
        Smooth CoV values following Viani et al. (2018).
        Smoothing window has size 0.66*bin_central^0.88
    '''
    smoothed_CoVs = [smoothing_func(center, bin_centers, CoVs) for center in bin_centers]
    return np.asarray(smoothed_CoVs)

def smoothing_func(center, bin_centers, CoVs):
    width = 0.66 * center**0.88
    lower = center - width / 2
    upper = center + width / 2
    indices = np.where((bin_centers >= lower) & (bin_centers < upper))[0]
    if len(indices) > 0:
        smoothed_val = np.nanmean(CoVs[indices])
    else:
        smoothed_val = np.nan
    return smoothed_val

def numax_estimate_CoV(bin_centers, smoothed_CoVs):
    '''
        Estimate numax as
    '''
    bin_centers_cut = bin_centers[bin_centers > 10]
    smoothed_CoVs_cut = smoothed_CoVs[bin_centers > 10]

    numax_init = bin_centers_cut[np.argmax(smoothed_CoVs_cut)]
    width = 0.66 * numax_init**0.88
    lower = numax_init - width / 2
    upper = numax_init + width / 2
    indices = np.where((bin_centers >= lower) & (bin_centers <= upper))[0]

    numerator = np.nansum(
        bin_centers[indices] * smoothed_CoVs[indices]
    )

    denominator = np.nansum(
        smoothed_CoVs[indices]
    )

    CoV_numax = numerator / denominator
    CoV_numax_error = np.nanstd(bin_centers[indices], ddof=1)
    print('numax_CoV:',CoV_numax, CoV_numax_error)
    return CoV_numax, CoV_numax_error