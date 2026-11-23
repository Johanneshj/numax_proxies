"""CoV scan: a power-excess sanity check of a fitted numax, independent of regularity.

The comb scan asks "is there a regular comb here?"; the CoV scan asks "is there excess
variance here?". It responds to oscillations whose comb the comb scan cannot see (very sharp
modes, mixed modes, missing modes), but also to artefacts and humps, so the two are read
together.

Around the guess numax:
  1. re-centre on the power excess (comb_scan.excess_peak: the maximum of P / median smoothed
     over one envelope FWHM within +- span FWHM) = the shifted numax
  2. windows of constant width `width` x Dnu (Dnu of the shifted numax), centres every Dnu / 4
     over the shifted numax +- span FWHM
  3. per window: CoV = std / mean of all n points of P / median (the spectrum normalised once
     at the start of the pipeline; no smoothing). White noise (chi^2_2): mean 1 - 1/n, sd
     1/sqrt(n) (checked by simulation), so z = (CoV - 1 + 1/n) sqrt(n)
  4. Gaussian (with offset, amplitude >= 0) fit to the z profile; significant if its peak
     z >= threshold; its centre is the CoV-scan numax.
"""
import numpy as np
from uncertainties import ufloat

from .comb_scan import STEP, _fit, excess_peak
from .seismic import delta_nu, envelope_fwhm


def cov_scan(frequency, rel_power, numax, width=3.0, span=2.0):
    """CoV scan around the power excess near the guess numax.

    Args:
        frequency: uniform frequency grid [muHz].
        rel_power: relative power P / running median - 1.
        numax: guess numax [muHz] (the ACF / CoV estimate).
        width: window width [Delta-nu of the shifted numax].
        span: re-centring range and scan range, +- span x envelope FWHM.

    Returns:
        dict (centres, cov, z, n, numax (the guess), scan_centre (the shifted numax), dnu,
        fwhm, width, popt, numax_cov (ufloat; NaN if no fit), peak_z, offset = numax_cov /
        numax - 1), or None if no window fits.
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
