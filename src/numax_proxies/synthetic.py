"""Synthetic power spectra with a known numax, for tests.

Granulation background (two Harvey profiles) plus white noise, and l = 0, 1, 2 Lorentzian
modes under a Gaussian envelope; multiplied by chi^2_2 noise. height_ratio = 0 gives noise only.
"""
import numpy as np

from .data import PSDData
from .seismic import delta_nu, envelope_fwhm

VISIBILITIES = {0: 1.0, 1: 1.5, 2: 0.6}


def kepler_lc_grid(years: float = 4.0, nyquist: float = 283.2) -> np.ndarray:
    """Frequency grid of a Kepler long-cadence PSD."""
    df = 1e6 / (years * 365.25 * 86400)
    return np.arange(df, nyquist, df)


def harvey(nu, a, b):
    """Harvey profile 2 sqrt(2) / pi a^2 / b / (1 + (nu / b)^4)."""
    return 2 * np.sqrt(2) / np.pi * a ** 2 / b / (1 + (nu / b) ** 4)


def background(nu, numax, white=1.0):
    """Granulation (two Harvey profiles) plus white noise."""
    a = 3382 * numax ** -0.609
    return white + harvey(nu, a, 0.317 * numax ** 0.970) + harvey(nu, a, 0.948 * numax ** 0.992)


def mode_spectrum(nu, numax, envelope_height, linewidth=0.1):
    """Lorentzian l = 0, 1, 2 modes under a Gaussian envelope."""
    dnu = float(delta_nu(numax))
    eps = 0.634 + 0.63 * np.log10(dnu)
    sigma_env = envelope_fwhm(numax) / (2 * np.sqrt(2 * np.log(2)))
    model = np.zeros_like(nu)
    n_lo = int((numax - 5 * sigma_env) / dnu) - 1
    n_hi = int((numax + 5 * sigma_env) / dnu) + 2
    vis_sum = sum(VISIBILITIES.values())
    for n in range(max(n_lo, 1), n_hi):
        nu0 = dnu * (n + eps)
        for l, offset in ((0, 0.0), (1, 0.5 * dnu), (2, -0.12 * dnu)):
            centre = nu0 + offset
            env = envelope_height * np.exp(-0.5 * ((centre - numax) / sigma_env) ** 2)
            # area of a Lorentzian = pi/2 * height * linewidth; areas per Dnu = env * Dnu
            height = 2 * env * dnu * VISIBILITIES[l] / (np.pi * linewidth * vis_sum)
            model += height / (1 + (2 * (nu - centre) / linewidth) ** 2)
    return model


def synthetic_psd(numax: float, height_ratio: float = 3.0, frequency=None, white: float = 1.0,
                  linewidth: float = 0.1, rng: np.random.Generator | None = None,
                  return_model: bool = False):
    """Noisy synthetic PSD with a known numax (and the noise-free model with return_model)."""
    rng = np.random.default_rng() if rng is None else rng
    nu = kepler_lc_grid() if frequency is None else np.asarray(frequency, dtype=float)
    bg = background(nu, numax, white)
    model = bg.copy()
    if height_ratio > 0:
        model += mode_spectrum(nu, numax, height_ratio * float(background(numax, numax, white)),
                               linewidth)
    psd = PSDData(nu, model * rng.exponential(size=nu.size))
    return (psd, model) if return_model else psd


def white_noise_psd(frequency=None, rng: np.random.Generator | None = None) -> PSDData:
    """Pure chi^2_2 white noise with unit mean."""
    rng = np.random.default_rng() if rng is None else rng
    nu = kepler_lc_grid() if frequency is None else np.asarray(frequency, dtype=float)
    return PSDData(nu, rng.exponential(size=nu.size))
