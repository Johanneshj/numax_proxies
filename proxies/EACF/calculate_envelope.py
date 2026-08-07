import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from matplotlib.colors import LogNorm
from scipy.signal import correlate, fftconvolve
import time as t

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

def calculate_envelope(
        frequency : NDArray,
        power : NDArray
):
    """
        Calculate envelope according to Mosser & Appourchaux 2009.
        They recommend a Hann window for the filtered spectrum.
    """
    df = np.mean(np.diff(frequency))
    width_muHz = 20
    hw = width_muHz / 2
    freq_grid = np.arange(1.0, np.max(frequency) + 5, 5)
    print(np.max(frequency))
    acfs = []
    envelopes = []
    lags = []
    for i, freq in enumerate(freq_grid):
        # Define power and frequency arrays in window
        indexes = np.where((frequency >= freq - hw) & (frequency <= freq + hw))[0] # indexes for window
        p = power[indexes] # power values in window
        f = frequency[indexes] # frequencies in window

        # Define hann window filter
        filt = hann(len(p))

        # Calculate product (essentially smoothing)
        prod = p * filt

        # Calculate inverse Fourier transform and grab positive frequencies
        ft = np.fft.ifft(prod)[:len(prod)//2]

        # Calculate normalized ACF as the real values
        # and the envelope as the absolute values
        acf = ft.real/ft.real[0]
        envelope = np.abs(ft)/np.abs(ft[0])

        # Define lag times in hours for plotting the autocorrelation
        tot = len(p) * df * 1e-6
        dt = 1 / (tot * 3600)
        lag = np.arange(len(acf)) * dt

        # Append values
        acfs.append(acf)
        envelopes.append(envelope)
        lags.append(lag)   

        if i == 40:
            print(dt)
            plt.figure()
            plt.plot(f, p/np.max(p), c='gray')
            plt.plot(f, prod/np.max(prod), c='k')
            plt.plot(f, filt, c='r')

            plt.figure()
            plt.plot(lag, acf, c='gray')
            plt.plot(lag, envelope, c='k')
            plt.plot(lag, -envelope, c='k')
            # plt.xlim(0, 100)
            # plt.vlines(69, ymin=-1, ymax=1, zorder=-1, colors='r', linestyles='--')

    return None

def gaussian(x : NDArray, A : float, sigma : float, mu : float):
    """Gaussian function"""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def get_bin_centers(frequency : NDArray, power : NDArray, 
                    overlap_scale : float = 6, min_num_points : int = 200, 
                    min_freq : float = 100, max_freq : float = None, width_factor : float = 1):
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

    # While loop generating bins
    if max_freq is None:
        max_freq = frequency[-1]

    while (bin_centers[-1] < max_freq):
        # Define next bin center and append
        next_center = bin_centers[-1] + (width_factor * 0.267 * bin_centers[-1] ** 0.764) / overlap_scale
        bin_centers.append(next_center)


    return bin_centers

    
def generate_filtered_matrix(
        frequency : NDArray,
        power : NDArray,
        bin_centers : NDArray
):
    # plt.figure()
    # plt.loglog(frequency, power)
    bin_centers = np.asarray(bin_centers)
    widths = 0.267 * bin_centers ** 0.764

    lefts = np.searchsorted(frequency, bin_centers - widths / 2)
    rights = np.searchsorted(frequency, bin_centers + widths / 2)

    filtered_power = [
        power[left:right] * 0.5 * (1 + np.cos(2 * np.pi * (frequency[left:right]-bin_center) / width))
        for left, right, bin_center, width in zip(lefts, rights, bin_centers, widths)
    ]

    plt.figure()
    for left, right, fp in zip(lefts[::5], rights[::5], filtered_power[::5]):
        plt.plot(frequency[left:right], fp, lw=0, ms=5, marker='.')
        plt.plot(frequency[left:right], power[left:right], c='k', zorder=-99)
    plt.xscale('log')
    return filtered_power

def two_dim_ACF(
        frequency : NDArray,
        power : NDArray,
):
    power, rel_filt = calculate_relative_power(frequency, power)
    
    bin_centers = get_bin_centers(frequency, power)
    filtered_power = generate_filtered_matrix(frequency, power, bin_centers)

    print('start 2d acf')
    start = t.time()
    acfs = [
        abs_acf(segment)
        for segment in filtered_power
    ]
    end = t.time()
    print(f'2d acf time = {np.round(end-start, 2)} seconds')

    CACF = [
        collapse_segment(seg) for seg in acfs
    ]
    # smoothed_acf = [
    #         smoothing_func(center, bin_centers, CACF) for center in bin_centers
    #     ]

    plt.figure()
    plt.plot(bin_centers, CACF, c='r')
    plt.xscale('log')
    # plt.plot(bin_centers, smoothed_acf)
    return None

def collapse_segment(seg : NDArray) -> float:
    """Collapse segment of total ACF and return mean."""
    # Ignore first index which always has ACF = 1
    seg = seg[1:]

    # Check length of segment
    if len(seg) < 1:
        return np.nan
    
    # Normalize by standard deviation in the case of logarithmically spaced bins
    std = np.nanstd(seg, ddof=1)
    if std == 0:
        return np.nan
    
    # Return collapsed acf
    mean = np.nanmean(seg)
    return mean

def smoothing_func(center : float, bin_centers : NDArray, ys : NDArray) -> float:
    """Smoothing function: calculate the mean in each bin."""
    width = 0.66 * center**0.88 # Smooth with FWHM of oscillation envelope
    # width = 0.267 * center**0.764
    lower = center - width / 4
    upper = center + width / 4
    indices = np.where((bin_centers >= lower) & (bin_centers <= upper))[0]
    if len(indices) > 0 and np.sum(~np.isnan(ys[indices])) > 0:
        smoothed_val = np.nanmean(ys[indices])
    else:
        smoothed_val = np.nan
    return smoothed_val

def abs_acf(x : NDArray):
    """
    Autocorrelation by means of np.correlate(x,x).
    Normalization tailored to log sliding windows

    Input:
        x :: PSD values

    Output:
        np.abs(corr) * scaling :: normalized absolute autocorrelation
    """
    # print(len(x), np.max(x))
    # Subtract mean
    x -= np.mean(x)  
    print(len(x))
    # Perform ACF on segment (x)
    corr = correlate(x, x, mode='full', method='fft')
    # corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]  # grab only the positive lags
    
    scaling = 1 #/ np.sqrt(len(x))
    # scaling = np.var(x) / np.sqrt(len(x))
    # print(len(x))

    return np.abs(corr) * scaling