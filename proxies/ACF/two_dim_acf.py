import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from .corr_acf_and_fft_acf import batch_fft_acf, abs_acf
import time as t

def calculate_two_dim_ACF(frequency=None, power=None, plot=False):

    '''
        Calculate 2D autocorrelation function:
            ACFs are calculated in sliding windows across entire spectrum.
            For 30-min cadence data, window size is 50 muHz and slides in increments of 10 muHz.
            Also, np.correlate is OK for 30-min cadence data.
            For shorter cadence data, window size is bigger and FFT based ACF is faster.
        
        Input:
            frequency :: list of frequencies in muHz
            power :: power normalized to relative power 
        
        Output:
            acf :: 2D AutoCorrelation Function
            freq_windows :: binned frequency list
                -> needed for plotting and collapsed ACF.
    '''
    # start = t.time()
    df = np.mean(np.diff(frequency))

    if np.max(frequency) > 300:
        window_size_muHz = 250  # window width
        overlap_muHz = 249      # overlap
    else:
        window_size_muHz = 50   # window width
        overlap_muHz = 45       # overlap
       
    window_size = int(window_size_muHz / df)
    step = int((window_size_muHz - overlap_muHz) / df)
    # print(step)
    if step <= 0:
        step = 10 # failsafe
    freq_windows = sliding_window_view(frequency, window_shape=window_size)[::step]
    power_windows = sliding_window_view(power, window_shape=window_size)[::step].copy()
    
    if np.max(frequency) > 500:
        acf = batch_fft_acf(power_windows)
    else:
        acf = np.array([abs_acf(seg) for seg in power_windows]) #
    # end = t.time()
    # print(f"2DACF computation time with averaged psd: {end - start:.4f} seconds")

    return acf, freq_windows 
    