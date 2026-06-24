import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

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
