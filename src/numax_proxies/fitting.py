"""Gaussian envelope models used by the ACF and CoV fits."""
import numpy as np

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))   # FWHM = 2.355 sigma


def gaussian(x, A, sigma, mu, y):
    """Gaussian with a free offset: y + A exp(-(x - mu)^2 / (2 sigma^2))."""
    return y + A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def gaussian_offset_one(x, A, sigma, mu):
    """Gaussian with offset 1: 1 + A exp(-(x - mu)^2 / (2 sigma^2))."""
    return 1.0 + A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def gaussian_jac(x, A, sigma, mu, y):
    """Analytic Jacobian of gaussian w.r.t. (A, sigma, mu, y), shape (len(x), 4)."""
    d = x - mu
    e = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return np.column_stack([e, A * e * d ** 2 / sigma ** 3, A * e * d / sigma ** 2,
                            np.ones_like(x)])


def gaussian_offset_one_jac(x, A, sigma, mu):
    """Analytic Jacobian of gaussian_offset_one w.r.t. (A, sigma, mu), shape (len(x), 3)."""
    d = x - mu
    e = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return np.column_stack([e, A * e * d ** 2 / sigma ** 3, A * e * d / sigma ** 2])
