"""Background correction of a fitted numax.

The ACF, the CoV and the comb scan respond to the oscillation envelope relative to the
background, H(nu) / B(nu). Where the background falls locally as a power law B ~ nu^-alpha,
a Gaussian envelope H at numax with width sigma gives a ratio that peaks above numax:

    d/dnu [ -(nu - numax)^2 / (2 sigma^2) + alpha ln nu ] = 0
    =>  nu_p = numax + alpha sigma^2 / nu_p.

Any statistic that increases with H/B peaks at the same nu_p, so

    numax = nu_p - alpha sigma^2 / nu_p,     sigma = FWHM(nu_p) / (2 sqrt(2 ln 2)),

with alpha measured from the running median B at nu_p -+ DISTANCE x FWHM (outside the
envelope): alpha = -ln(B(nu+) / B(nu-)) / ln(nu+ / nu-). Where white noise dominates,
alpha ~ 0 and the correction vanishes.
"""
import numpy as np

from .fitting import FWHM_TO_SIGMA
from .seismic import envelope_fwhm

DISTANCE = 1.5          # alpha from the running median at nu_p -+ DISTANCE x envelope FWHM
MIN_LEVER = 1.1         # need nu+ / nu- >= MIN_LEVER (else no correction)


def background_slope(frequency, median, nu) -> float:
    """Local power-law index alpha = -d ln B / d ln nu of the running median around nu.

    Args:
        frequency: frequency grid [muHz].
        median: running median of the PSD (the background) on that grid.
        nu: frequency [muHz].

    Returns:
        alpha, or NaN if the lever arm inside the spectrum is too short.
    """
    fw = float(envelope_fwhm(nu))
    lo = max(nu - DISTANCE * fw, frequency[0])
    hi = min(nu + DISTANCE * fw, frequency[-1])
    if not (0 < lo < nu < hi) or hi / lo < MIN_LEVER:
        return np.nan
    b_lo, b_hi = np.interp([lo, hi], frequency, median)
    if not (b_lo > 0 and b_hi > 0):
        return np.nan
    return float(-np.log(b_hi / b_lo) / np.log(hi / lo))


def correct_numax(frequency, median, nu) -> float:
    """numax = nu - alpha sigma^2 / nu (unchanged if alpha cannot be measured).

    Args:
        frequency: frequency grid [muHz].
        median: running median of the PSD on that grid.
        nu: fitted numax [muHz].

    Returns:
        Corrected numax [muHz] (NaN stays NaN).
    """
    if not np.isfinite(nu) or nu <= 0:
        return nu
    alpha = background_slope(frequency, median, nu)
    if not np.isfinite(alpha):
        return nu
    sigma = float(envelope_fwhm(nu)) / FWHM_TO_SIGMA
    return nu - alpha * sigma ** 2 / nu
