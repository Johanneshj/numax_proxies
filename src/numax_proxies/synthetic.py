"""Synthetic power spectra with a known numax, for injection tests.

Model (all in muHz, power in arbitrary units):
    P(nu) = [ W + granulation(nu) + modes(nu) ] * chi^2_2 / 2

  granulation : two Harvey profiles with Kallinger+2014 scalings,
                b1 = 0.317 numax^0.970, b2 = 0.948 numax^0.992, a = 3382 numax^-0.609
  envelope    : Gaussian at numax with FWHM 0.66 numax^0.88; its height is
                height_ratio x (granulation + W) at numax
  modes       : radial orders n with nu_{n,0} = Delta-nu (n + eps), eps = 0.634 + 0.63 log10 Dnu
                (Mosser+2011); l=2 at nu_{n,0} - 0.12 Dnu, l=1 at nu_{n,0} + Dnu/2
                (no mixed modes); visibilities 1 : 1.5 : 0.6; Lorentzians of width
                `linewidth`. Mode heights are set so that the power averaged over one
                Delta-nu follows the Gaussian envelope.
  W           : white noise level

Set height_ratio = 0 for a pure noise (no oscillations) spectrum.
"""
import numpy as np

from .data import PSDData
from .seismic import delta_nu, envelope_fwhm

VISIBILITIES = {0: 1.0, 1: 1.5, 2: 0.6}


def kepler_lc_grid(years: float = 4.0, nyquist: float = 283.2) -> np.ndarray:
    """Frequency grid of a Kepler long-cadence PSD.

    Args:
        years: length of the time series [years].
        nyquist: Nyquist frequency [muHz].

    Returns:
        Frequency grid [muHz].
    """
    df = 1e6 / (years * 365.25 * 86400)
    return np.arange(df, nyquist, df)


def harvey(nu, a, b):
    """Harvey profile 2 sqrt(2) / pi a^2 / b / (1 + (nu / b)^4)."""
    return 2 * np.sqrt(2) / np.pi * a ** 2 / b / (1 + (nu / b) ** 4)


def background(nu, numax, white=1.0):
    """Granulation (two Kallinger+2014 Harvey profiles) plus white noise.

    Args:
        nu: frequency [muHz].
        numax: [muHz].
        white: white-noise level.

    Returns:
        Background power.
    """
    a = 3382 * numax ** -0.609
    return white + harvey(nu, a, 0.317 * numax ** 0.970) + harvey(nu, a, 0.948 * numax ** 0.992)


def mode_spectrum(nu, numax, envelope_height, linewidth=0.1):
    """Lorentzian l = 0, 1, 2 modes under a Gaussian envelope.

    Args:
        nu: frequency [muHz].
        numax: [muHz].
        envelope_height: envelope height (power averaged over one Delta-nu).
        linewidth: Lorentzian width [muHz].

    Returns:
        Mode power.
    """
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
    """Noisy synthetic PSD with a known numax.

    Args:
        numax: [muHz].
        height_ratio: envelope height / background at numax (0 = no oscillations).
        frequency: frequency grid [muHz]; None = Kepler LC, 4 years.
        white: white-noise level.
        linewidth: mode width [muHz].
        rng: numpy random generator.
        return_model: also return the noise-free model.

    Returns:
        PSDData, or (PSDData, model).
    """
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
    """Pure chi^2_2 white noise with unit mean.

    Args:
        frequency: frequency grid [muHz]; None = Kepler LC, 4 years.
        rng: numpy random generator.

    Returns:
        PSDData.
    """
    rng = np.random.default_rng() if rng is None else rng
    nu = kepler_lc_grid() if frequency is None else np.asarray(frequency, dtype=float)
    return PSDData(nu, rng.exponential(size=nu.size))
