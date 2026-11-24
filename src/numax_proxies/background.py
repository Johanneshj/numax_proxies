"""Background correction of a fitted numax.

The methods respond to the oscillation power relative to the background, which peaks slightly
above numax where the background falls with frequency. correct_numax shifts the fitted value
back, using the local slope of the background.
"""
import numpy as np

from .fitting import FWHM_TO_SIGMA
from .seismic import envelope_fwhm

DISTANCE = 1.5          # alpha from the running median at nu_p -+ DISTANCE x envelope FWHM
MIN_LEVER = 1.1         # need nu+ / nu- >= MIN_LEVER (else no correction)


def background_slope(frequency, median, nu) -> float:
    """Local slope alpha of the background (B ~ nu^-alpha) from the running median on both
    sides of nu. NaN if the two points are too close together (MIN_LEVER).
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
    """Background-corrected numax: nu - alpha sigma^2 / nu (unchanged if alpha is NaN)."""
    if not np.isfinite(nu) or nu <= 0:
        return nu
    alpha = background_slope(frequency, median, nu)
    if not np.isfinite(alpha):
        return nu
    sigma = float(envelope_fwhm(nu)) / FWHM_TO_SIGMA
    return nu - alpha * sigma ** 2 / nu
