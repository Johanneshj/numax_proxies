"""CoV scan: checks that there is excess power (not necessarily regular) around a fitted numax.

Re-centres on the power excess, computes the CoV of P / median in windows of constant width,
converts it to z against white noise and fits a Gaussian. Significant if the peak z is above
the threshold; the fitted centre is the CoV-scan numax.
"""
import numpy as np
from uncertainties import ufloat

from .comb_scan import STEP, _fit, excess_peak
from .seismic import delta_nu, envelope_fwhm


def cov_scan(frequency, rel_power, numax, width=3.0, span=2.0):
    """Run the CoV scan around numax.

    Returns a dict with the windows (centres, cov, z), the shifted numax (scan_centre), the fit
    (popt, numax_cov, peak_z), or None if the scan does not fit in the spectrum.
    """
    frequency = np.asarray(frequency, dtype=float)
    rel_power = np.asarray(rel_power, dtype=float)
    if not np.isfinite(numax) or numax <= 0:
        return None
    guess = float(numax)
    shifted = excess_peak(frequency, rel_power, guess, span)
    if not np.isfinite(shifted):
        return None
    dnu, fwhm = float(delta_nu(shifted)), float(envelope_fwhm(shifted))
    df = frequency[1] - frequency[0]
    n = int(round(width * dnu / df))
    if n < 10:
        return None
    centres = np.arange(shifted - span * fwhm, shifted + span * fwhm + 1e-9, STEP * dnu)
    starts = np.searchsorted(frequency, centres - 0.5 * width * dnu)
    ok = (centres - 0.5 * width * dnu >= frequency[0]) & (starts + n <= frequency.size)
    centres, starts = centres[ok], starts[ok]
    if centres.size < 5:
        return None
    p_med = rel_power + 1.0                                  # P / median
    cov = np.array([p_med[a:a + n].std(ddof=1) / p_med[a:a + n].mean() for a in starts])
    z = (cov - 1.0 + 1.0 / n) * np.sqrt(n)
    p, pcov = _fit(centres, z, shifted, fwhm)
    centre = ufloat(p[2], np.sqrt(pcov[2, 2])) if p is not None else ufloat(np.nan, np.nan)
    return dict(centres=centres, cov=cov, z=z, n=n, numax=guess, scan_centre=shifted, dnu=dnu,
                fwhm=fwhm, width=float(width), popt=p, numax_cov=centre,
                peak_z=np.nan if p is None else float(p[0] + p[3]),
                offset=centre.n / guess - 1.0)


def significant(r, threshold) -> bool:
    """True if the scan has a positive Gaussian with peak z >= threshold."""
    return (r is not None and r["popt"] is not None and r["popt"][0] > 0
            and r["peak_z"] >= threshold)
