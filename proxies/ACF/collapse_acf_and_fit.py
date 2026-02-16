import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings


def collapsed_acf(acf=None, freq_windows=None, *args, **kwargs):
    """
    Collapse 2D ACF into 1D ACF

    Input:
        acf :: 2D ACF
        freq_windows :: binned frequency list

    Output:
        collapsed_acf_numax :: collapsed 1D acf
        freq_centers :: medians of freq_windows for plotting and fitting
    """

    collapsed_acf_numax = np.mean(acf, axis=1)
    collapsed_acf_numax = collapsed_acf_numax - np.median(collapsed_acf_numax)
    # collapsed_acf_numax = (collapsed_acf_numax - np.min(collapsed_acf_numax))/(np.max(collapsed_acf_numax)+np.min(collapsed_acf_numax))
    collapsed_acf_numax /= np.max(collapsed_acf_numax)
    freq_centers = np.median(freq_windows, axis=1)

    return collapsed_acf_numax, freq_centers


def fit_gauss_to_collapsed_acf(collapsed_acf_numax=None, freq_centers=None):
    """
    Fit Gaussian to collapsed ACF:
        First estimates:
            - width = (2/3) * numax ** (22/25) (Mosser+ 2012)
            - Amplitude = 1 (because regularized)
            - mean = numax = "freq at maximum" (first numax guess)

    Input:
        collapsed_acf_numax :: collapsed 1D acf
        freq_centers :: medians of freq_windows for plotting and fitting

    Output:
        numax :: numax estimate (central value of Gauss) in muHz
    """

    # idx = np.where((freq_centers > 10) & (freq_centers < 5000))[0]
    idx_max = np.argmax(collapsed_acf_numax[1:])
    numax = freq_centers[idx_max]
    # print('p0:', numax)

    collapsed_acf_numax = collapsed_acf_numax - np.median(collapsed_acf_numax)
    # collapsed_acf_numax = (collapsed_acf_numax - np.min(collapsed_acf_numax))/(np.max(collapsed_acf_numax)+np.min(collapsed_acf_numax)) # Regularizing
    collapsed_acf_numax /= np.max(collapsed_acf_numax)

    window = 4 * (2 / 3) * numax ** (22 / 25)  # Mosser
    mask = (freq_centers > numax - window) & (freq_centers < numax + window)

    x = freq_centers[mask]
    y = collapsed_acf_numax[mask]

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    def gaussian(x, A, sigma, mu):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    # We do a "try" here in case fits fails or is underresolved
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)

            popt, pcov = curve_fit(
                gaussian,
                freq_centers[mask],
                collapsed_acf_numax[mask],
                p0=[1, (2 / 3) * numax ** (22 / 25), numax],
            )

        numax = popt[2]
        return numax, popt

    except (RuntimeError, OptimizeWarning, ValueError):
        return np.nan, np.nan
