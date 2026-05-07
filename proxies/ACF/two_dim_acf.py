import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .corr_acf_and_fft_acf import batch_fft_acf, abs_acf, abs_acf_linear
import time as t
import matplotlib.pyplot as plt


def calculate_two_dim_ACF(frequency=None, power=None, plot=False, 
                          sliding_window_flag='log_numax', acf_params={}):
    """
    Calculate 2D autocorrelation function:
        Three options are provided:
        1. Linear sliding window (Huber+ 2009, Viania+ 2019):
            This method works ok, but fails at lower frequencies
        2. Logarithmic sliding window (this work):
            This method can give OK results, but is very dependent on tuning bin sizes and width, 
            which complicates things
        3. Other method (Viani+ 2019):
            Use a sliding window where each window slides a distance equal to some fraction of a 
            potential delta_nu at the center of the previous bin.
            Each bin is sized as if a potential delta_nu could fit in it.
        
        ===> For now, method 3 seems to work best.

    Input:
        frequency :: list of frequencies in muHz
        power :: power normalized to relative power

    Output:
        acf :: 2D AutoCorrelation Function
        freq_windows :: binned frequency list
            -> needed for plotting and collapsed ACF.
    """
    
    
    start = t.time()
    # Check flags
    if sliding_window_flag == 'linear':
        # Linear sliding window
        freq_windows, power_windows = linear_sliding_window(
            frequency=frequency,
            power=power
        )
        # Calculate acf for each segment
        if np.max(frequency) > 300:
            # Long cadence data
            acf = batch_fft_acf(power_windows)
        else:
            # Short cadence data
            acf = np.array([abs_acf_linear(seg) for seg in power_windows])
    elif sliding_window_flag == 'log':
        # Log sliding window
        freq_windows, power_windows = log_sliding_window(
            frequency=frequency,
            power=power
        )
        # Calculate acf for each segment
        acf = [abs_acf(seg) for seg in power_windows]
    elif sliding_window_flag == 'log_numax':
        # Special sliding window (Viani et al. 2019)
        # Check if we defined some of the sliding window parameters our selves:
        overlap_scale = acf_params.get('ACF_overlap_scale')
        min_num_points = acf_params.get('ACF_min_num_points')
        min_freq = acf_params.get('ACF_min_freq')
        print(min_freq)
        width_factor = acf_params.get('ACF_width_factor')
        # Grab the remaining with binning_parameters function
        overlap_scale, min_num_points, min_freq, width_factor = binning_parameters(frequency, overlap_scale, min_num_points, min_freq, width_factor)
        freq_windows, power_windows = other_binning(
            frequency=frequency,
            power=power,
            overlap_scale=overlap_scale,
            min_num_points=min_num_points,
            min_freq=min_freq,
            max_freq=None,
            width_factor=width_factor
        )
        # Calculate acf for each segment
        acf = [abs_acf(seg) for seg in power_windows]

    end = t.time()
    print(f'2D ACF calculation time: {np.round(end-start, 2)} seconds')

    return acf, freq_windows

def binning_parameters(frequency, overlap_scale=None, min_num_points=None, min_freq=None, width_factor=None):
    """
        Sliding window parameters for log_numax sliding window (Viani+ 2019).
        This parameters have been determined by trial-and-error.
    """
    max_freq = np.max(frequency)
    if max_freq > 300:
        # Short cadence data
        if max_freq > 5000:
            # Kepler 1 minute cadence
            overlap_scale = 2 if overlap_scale is None else overlap_scale
            min_num_points = 50 if min_num_points is None else min_num_points
            min_freq = 100 if min_freq is None else min_freq
            width_factor = 1 if width_factor is None else width_factor
        else:
            # TESS 2 minute cadence
            overlap_scale = 2 if overlap_scale is None else overlap_scale
            min_num_points = 30 if min_num_points is None else min_num_points
            min_freq = 10 if min_freq is None else min_freq
            width_factor = 1 if width_factor is None else width_factor
    else:
        # 30 minute cadence data
        overlap_scale = 10 if overlap_scale is None else overlap_scale
        min_num_points = 20 if min_num_points is None else min_num_points
        min_freq = 1 if min_freq is None else min_freq
        width_factor = 1 if width_factor is None else width_factor
    return overlap_scale, min_num_points, min_freq, width_factor

def log_sliding_window(frequency, power, overlap_scale=2):
    '''Log sliding window - testing has revealed that log_numax generally performs better'''
    # Define bin centers in log space
    bin_centers, bin_widths = get_bins(frequency=frequency)
    # Lists we append results too
    freq_windows = []
    power_windows = []
    for bin_center, bin_width in zip(bin_centers, bin_widths):
        # Overlap controlled by overlap_scale, here we define mask
        mask = (frequency >= bin_center - bin_width / overlap_scale) & \
                (frequency <= bin_center + bin_width / overlap_scale)
        frequency_window = frequency[mask]
        power_window = power[mask]
        if len(power_window) > 1:
            freq_windows.append(frequency_window)
            power_windows.append(power_window)
    return freq_windows, power_windows

def bin_centers_and_widths(frequency):
    """
        Bin centers are widths for log sliding window.
        These values are found by trial-and-error, in general,
        log_numax sliding window performs better.
    """
    nu_nyq = np.max(frequency)
    # For short cadence we cut the spectrum from 100 muHz,
    # which isn't perfect
    print('length of spec:', len(frequency))
    print('resolution:', np.mean(np.diff(frequency)))
    if nu_nyq > 5000: # Short cadence
        bin_centers = np.geomspace(
            1,
            np.max(frequency),
            10000
        )
        # Bin widths
        bin_widths = np.geomspace(
            10 * np.mean(np.diff(frequency)), # 10 data points
            300,
            10000
        )
    elif (nu_nyq < 5000) and (nu_nyq > 300):
        # TESS short cadence
        bin_centers = np.geomspace(
            1,
            np.max(frequency),
            2000
        )
        # Bin widths
        bin_widths = np.geomspace(
            0.5,
            200,
            2000
        )
    else: # long cadence
        bin_centers = np.geomspace(
            np.min(frequency),
            np.max(frequency),
            int(len(frequency)/10)
        )
        bin_widths = np.geomspace(
            10 * np.mean(np.diff(frequency)),
            10000 * np.mean(np.diff(frequency)),
            int(len(frequency)/10)
        )
    return bin_centers, bin_widths

def linear_sliding_window(frequency, power):
    """Linear sliding window (Viani+ 2019)"""
    df = np.mean(np.diff(frequency))

    if np.max(frequency) > 300:
        window_size_muHz = 250  # window width
        overlap_muHz = 249  # overlap
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

def other_binning(frequency, power, overlap_scale=2, min_num_points=200, min_freq=100, max_freq=None, width_factor=1):
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
    # Define fail-safe width
    df = np.mean(np.diff(frequency))
    width_floor = min_num_points * df

    # Define initial bin center and initial width
    init_bin_center = np.max([frequency[0], min_freq])
    bin_centers = [init_bin_center]
    init_width = np.max([width_factor * 0.267 * bin_centers[0] ** 0.764, width_floor])
    bin_widths = [init_width]

    # While loop generating bins
    if max_freq is None:
        max_freq = frequency[-1]

    while (bin_centers[-1] < max_freq):
        # Define next bin center and append
        next_center = bin_centers[-1] + (width_factor * 0.267 * bin_centers[-1] ** 0.764) / overlap_scale
        bin_centers.append(next_center)

        # Define next bin width and append
        width = np.max([width_factor * 0.267 * next_center**0.764, width_floor])
        bin_widths.append(width)

    # Appending frequency and power values of each bin
    fs = []
    ps = []
    for center, width in zip(bin_centers, bin_widths):
        f, p = other_binning_freqs_power(center, width, frequency, power)
        fs.append(f)
        ps.append(p)

    return fs, ps

def other_binning_freqs_power(center, width, frequency, power):
    """
    Grab frequency and power values in bins defined with other_binning function.

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
    index_bin = np.where((frequency >= lower) & (frequency <= upper))[0]

    # Is bin length zero?
    if len(index_bin) == 0:
        return np.nan

    return frequency[index_bin], power[index_bin]

def get_bins(frequency, min_points=20):
    """Use frequency array to get the optimal geometric windows"""
    f = frequency # frequency array
    df = np.mean(np.diff(frequency)) # Resolution
    # min_points = 20 # minimum number of data points in first bin

     # Define bin centers
    bin_centers = np.geomspace(f[10], f[-10], 1000)

    # Define bin widths
    start_w = min_points * df
    end_w  = (f[-1]-f[0])*0.1
    bin_widths = np.geomspace(start_w, end_w, len(bin_centers))

    return bin_centers, bin_widths