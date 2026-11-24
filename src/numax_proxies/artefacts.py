"""Kepler short-cadence artefacts (STAR.cadence == "Kepler_SC").

artefact_cleaning: replace a listed artefact by noise only where a significant narrow peak is
found. comb_check: require a significant PSPS peak near 1/Delta-nu or 2/Delta-nu of a fitted numax.
"""
import logging

import numpy as np

from .binning import running_median
from .seismic import delta_nu, envelope_fwhm

logger = logging.getLogger(__name__)

# Van Cleve et al. (2016, Kepler Data Characteristics Handbook), SC spurious frequency summary:
# (frequency [muHz], width class). "broad" = Broad / V.broad; "narrow" = Narrow or not given.
KEPLER_SC_ARTEFACTS = [
    (86.8, "broad"), (196.5, "broad"), (242.5, "broad"), (290.0, "broad"), (340.0, "broad"),
    (344.9, "broad"), (360.0, "narrow"), (362.8, "broad"), (365.8, "broad"), (370.4, "narrow"),
    (421.0, "narrow"), (566.4, "narrow"), (1132.8, "narrow"), (1680.1, "narrow"),
    (1699.2, "narrow"), (1803.4, "narrow"), (1825.2, "narrow"), (2092.6, "narrow"),
    (2265.6, "narrow"), (2832.0, "narrow"), (3398.3, "narrow"), (3964.7, "narrow"),
    (4428.4, "narrow"), (4451.4, "narrow"), (4531.1, "narrow"), (4995.8, "narrow"),
    (5016.2, "narrow"), (5097.5, "narrow"), (5564.4, "broad"), (5584.5, "narrow"),
    (5663.9, "narrow"), (6230.3, "narrow"), (6796.7, "narrow"), (7024.0, "narrow"),
    (7363.1, "narrow"), (7444.0, "narrow"), (7865.0, "narrow"), (7929.5, "narrow"),
    (8286.0, "narrow"), (8495.9, "narrow"),
]


def max_exceedance_threshold(n_points: int, fap: float) -> float:
    """Threshold that the largest of n_points Exp(1) values exceeds with probability fap."""
    return float(-np.log(-np.expm1(np.log1p(-fap) / max(n_points, 1))))


# --------------------------------------------------------------------------------------
# Option 1: detection-based cleaning
# --------------------------------------------------------------------------------------
def clean_detected_artefacts(frequency, power, rng, fap=0.01, search_halfwidth=1.0,
                             max_halfwidth=3.0, broad_factor=5.0, clip=3.0,
                             artefacts=KEPLER_SC_ARTEFACTS, med=None):
    """Replace listed artefacts by noise where a significant narrow peak is found.

    Returns (cleaned power, report per artefact).
    """
    frequency = np.asarray(frequency, dtype=float)
    power = np.array(power, dtype=float, copy=True)
    df = frequency[1] - frequency[0]
    med = running_median(frequency, power) if med is None else med
    x = power / med * np.log(2)
    report = []
    for nu0, width_class in artefacts:
        f = broad_factor if width_class == "broad" else 1.0
        sel = np.flatnonzero(np.abs(frequency - nu0) <= f * search_halfwidth)
        if sel.size == 0:
            continue
        thr = max_exceedance_threshold(sel.size, fap)
        j = sel[np.argmax(x[sel])]
        entry = dict(frequency=nu0, detected=bool(x[j] > thr), peak=frequency[j],
                     x_max=float(x[j]), threshold=thr, lo=np.nan, hi=np.nan)
        if entry["detected"]:
            reach = int(np.ceil(f * max_halfwidth / df))
            lo = j
            while lo > max(0, j - reach) and x[lo - 1] > clip:
                lo -= 1
            hi = j
            while hi < min(len(x) - 1, j + reach) and x[hi + 1] > clip:
                hi += 1
            power[lo:hi + 1] = rng.exponential(scale=float(np.mean(med[lo:hi + 1])) / np.log(2),
                                               size=hi - lo + 1)
            entry.update(lo=frequency[lo], hi=frequency[hi])
        report.append(entry)
    n = sum(e["detected"] for e in report)
    logger.info("Kepler SC artefacts: %d of %d listed frequencies detected and cleaned",
                n, len(report))
    return power, report


def replace_regions(frequency, power, regions, rng, med=None):
    """Copy of the PSD with fixed frequency regions replaced by noise at the local level."""
    frequency = np.asarray(frequency, dtype=float)
    power = np.array(power, dtype=float, copy=True)
    if not regions:
        return power
    med = running_median(frequency, power) if med is None else med
    for lo, hi in regions:
        m = (frequency >= lo) & (frequency <= hi)
        if m.any():
            power[m] = rng.exponential(scale=float(np.median(med[m])) / np.log(2), size=m.sum())
    return power


# --------------------------------------------------------------------------------------
# Option 2: comb check with the power spectrum of the power spectrum
# --------------------------------------------------------------------------------------
def near_artefact(numax: float, distance: float = 0.25, artefacts=KEPLER_SC_ARTEFACTS) -> bool:
    """True if a listed artefact is within distance x envelope FWHM of numax."""
    if not np.isfinite(numax) or numax <= 0:
        return False
    return any(abs(nu0 - numax) <= distance * envelope_fwhm(numax) for nu0, _ in artefacts)


def psps(frequency, rel_power, numax, tolerance=0.3, fap=0.01, n_fwhm=1.0, oversampling=1):
    """Power spectrum of the power spectrum around numax: comb test and Delta-nu.

    Returns a dict with the PSPS (t, z, oversampled t_fine, z_fine), the threshold, comb (bool) and
    spacing (Delta-nu estimate); None if the stretch of spectrum is too short.
    """
    if not np.isfinite(numax) or numax <= 0:
        return None
    sel = np.abs(frequency - numax) <= n_fwhm * envelope_fwhm(numax)
    x = np.asarray(rel_power, dtype=float)[sel]
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return None
    x = x - x.mean()
    var = np.mean(x * x)
    if var <= 0:
        return None
    df = frequency[1] - frequency[0]
    z = np.abs(np.fft.rfft(x)) ** 2 / (n * var)
    t = np.fft.rfftfreq(n, d=df)                              # [1/muHz]
    dnu = float(delta_nu(numax))
    search = np.zeros(t.size, dtype=bool)
    for m in (1.0, 2.0):
        search |= (t >= m / (dnu * (1 + tolerance))) & (t <= m / (dnu * (1 - tolerance)))
    search[0] = False
    if not search.any():
        return None
    k = np.flatnonzero(search)[np.argmax(z[search])]
    # TODO: work out the PSPS detection threshold properly. Now: z ~ Exp(1) per independent
    # t (white, Gaussian x assumed) and the max-of-M threshold over the search points. Open:
    # x = P / median - 1 is chi^2-like and the running median correlates neighbouring
    # points; the search ranges around 1/Dnu and 2/Dnu are not independent of the comb.
    thr = max_exceedance_threshold(int(search.sum()), fap)
    # oversampled PSPS (zero padding): same normalisation, equal to z at the original t
    n_pad = n * max(int(oversampling), 1)
    z_fine = np.abs(np.fft.rfft(x, n_pad)) ** 2 / (n * var)
    t_fine = np.fft.rfftfreq(n_pad, d=df)
    fine = np.zeros(t_fine.size, dtype=bool)
    for m in (1.0, 2.0):
        fine |= (t_fine >= m / (dnu * (1 + tolerance))) & (t_fine <= m / (dnu * (1 - tolerance)))
    kf = np.flatnonzero(fine)[np.argmax(z_fine[fine])]
    spacing = 1.0 / t_fine[kf] * (2.0 if t_fine[kf] > 1.5 / dnu else 1.0)   # back to Delta-nu
    return dict(t=t, z=z, search=search, threshold=thr, dnu=dnu, z_max=float(z[k]),
                comb=bool(z[k] > thr), t_fine=t_fine, z_fine=z_fine, spacing=float(spacing))


class CombChecker:
    """numax -> passes the comb check? (cached; "near_artefacts" tests only fits near a listed artefact)."""

    def __init__(self, frequency, rel_power, mode="near_artefacts", tolerance=0.3, fap=0.01,
                 distance=0.25):
        self.frequency = np.asarray(frequency, dtype=float)
        self.rel_power = np.asarray(rel_power, dtype=float)
        self.mode, self.tolerance, self.fap, self.distance = mode, tolerance, fap, distance
        self._cache = {}

    def __call__(self, numax: float) -> bool:
        """True if numax is accepted (no test needed, or a comb is found)."""
        if not np.isfinite(numax) or numax <= 0:
            return False
        if self.mode == "near_artefacts" and not near_artefact(numax, self.distance):
            return True
        key = int(round(np.log(numax) * 1000))                 # 0.1 % steps in numax
        if key not in self._cache:
            r = psps(self.frequency, self.rel_power, float(np.exp(key / 1000)),
                     self.tolerance, self.fap)
            self._cache[key] = r is not None and r["comb"]
        return self._cache[key]
