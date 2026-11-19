"""Gaussian envelope models used by the ACF and CoV fits."""
import numpy as np

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))   # FWHM = 2.355 sigma


def gaussian(x, A, sigma, mu, y):
    """Gaussian with a free offset: y + A exp(-(x - mu)^2 / (2 sigma^2))."""
    return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def gaussian_offset_one(x, A, sigma, mu):
    """Gaussian with offset 1: 1 + A exp(-(x - mu)^2 / (2 sigma^2))."""
    return 1.0 + A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
