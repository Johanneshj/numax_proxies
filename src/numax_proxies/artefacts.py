"""Kepler short-cadence artefacts: detection-based cleaning and a comb check of fitted numax.

Both are used only for STAR.cadence == "Kepler_SC" (switches in CONFIG):
  artefact_cleaning : at each listed artefact frequency, clean the spectrum only if a
                      significant narrow peak is found there (the oscillation comb survives).
  comb_check        : after a fit, require a regular pattern (a comb with spacing ~Delta-nu)
                      around the fitted numax, tested with the power spectrum of the power
                      spectrum (PSPS); "near_artefacts" (only fits near a listed artefact) or
                      "always".
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
    """Threshold u such that the largest of n_points independent Exp(1) values exceeds u
    with probability fap: P(max > u) = 1 - (1 - exp(-u))^n."""
    return float(-np.log(-np.expm1(np.log1p(-fap) / max(n_points, 1))))


# --------------------------------------------------------------------------------------
# Option 1: detection-based cleaning
# --------------------------------------------------------------------------------------
def clean_detected_artefacts(frequency, power, rng, fap=0.01, search_halfwidth=1.0,
                             max_halfwidth=3.0, broad_factor=5.0, clip=3.0,
                             artefacts=KEPLER_SC_ARTEFACTS):
    """Clean listed artefacts only where a significant narrow peak is present.

    For each artefact nu0: the PSD is divided by its running median and expressed in units
    of the mean (x = P / median * ln 2, ~Exp(1) for noise). If the largest x within
    +- search_halfwidth of nu0 exceeds the max-of-M threshold at fap, the contiguous points
    around that maximum with x > clip (at most +- max_halfwidth around it) are replaced by
    chi^2_2 noise with the local mean power. Broad artefacts use broad_factor x both widths.

    Args:
        frequency: frequency grid [muHz] (uniform).
        power: PSD.
        rng: numpy random generator.
        fap: false-alarm probability of the detection, per artefact.
        search_halfwidth: [muHz] search window around each listed frequency.
        max_halfwidth: [muHz] largest half-width that is replaced.
        broad_factor: widths multiplied by this for broad artefacts.
        clip: [mean power] the replaced region extends while x > clip.
        artefacts: [(frequency [muHz], "narrow" | "broad")].

    Returns:
        (cleaned power, report: list of dicts with frequency, detected, peak [muHz],
        x_max, threshold, lo, hi [muHz]).
    """
    frequency = np.asarray(frequency, dtype=float)
    power = np.array(power, dtype=float, copy=True)
    df = frequency[1] - frequency[0]
    med = running_median(frequency, power)
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


# --------------------------------------------------------------------------------------
# Option 2: comb check with the power spectrum of the power spectrum
# --------------------------------------------------------------------------------------
def near_artefact(numax: float, distance: float = 0.25, artefacts=KEPLER_SC_ARTEFACTS) -> bool:
    """Is a listed artefact within distance x envelope FWHM of numax?"""
    if not np.isfinite(numax) or numax <= 0:
        return False
    return any(abs(nu0 - numax) <= distance * envelope_fwhm(numax) for nu0, _ in artefacts)


def psps(frequency, rel_power, numax, tolerance=0.3, fap=0.01, n_fwhm=1.0, oversampling=1):
    """Power spectrum of the power spectrum (PSPS) around numax, and the comb test on it.

    The relative power x = P / median - 1 within +- n_fwhm envelope FWHM of numax is
    centred and Fourier transformed; z(t) = |FT_t|^2 / (N var(x)) is ~Exp(1) at every t > 0
    for white noise. A comb with spacing Delta-nu gives peaks at t = 1 / Delta-nu and (from
    the Delta-nu / 2 spacing of l = 0 and l = 1) at 2 / Delta-nu; a single spike gives a flat
    PSPS. The largest z at t within +- tolerance of 1/Delta-nu(numax) or 2/Delta-nu(numax) is
    compared with the max-of-M threshold at fap.

    Args:
        frequency: frequency grid [muHz] (uniform).
        rel_power: P / median - 1 on that grid.
        numax: [muHz].
        tolerance: relative search range around 1/Delta-nu and 2/Delta-nu.
        fap: false-alarm probability of the test.
        n_fwhm: half-width of the stretch of spectrum, in envelope FWHM.
        oversampling: zero-padding factor for t_fine / z_fine (plot and spacing only; the
            test uses the independent points t, z).

    Returns:
        dict with t [1/muHz], z (independent points), search (bool mask), threshold,
        dnu (Delta-nu(numax)), z_max, comb (bool), t_fine, z_fine (oversampled PSPS) and
        spacing (Delta-nu of the strongest oversampled peak in the search ranges [muHz]);
        None if the stretch is too short.
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


def psps_comb_test(frequency, rel_power, numax, tolerance=0.3, fap=0.01, n_fwhm=1.0):
    """Is there a comb with spacing ~Delta-nu(numax) around numax? (see psps)

    Returns:
        (comb present, z_max, spacing of the strongest PSPS peak [muHz]).
    """
    r = psps(frequency, rel_power, numax, tolerance, fap, n_fwhm)
    if r is None:
        return False, np.nan, np.nan
    return r["comb"], r["z_max"], r["spacing"]


class CombChecker:
    """numax -> passes the comb check? (results cached on numax in 0.1 % steps).

    Args:
        frequency: frequency grid [muHz].
        rel_power: P / median - 1.
        mode: "near_artefacts" (test only fits within distance x envelope FWHM of a listed
            artefact) or "always".
        tolerance, fap: see psps_comb_test.
        distance: see near_artefact.
    """

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
            self._cache[key] = psps_comb_test(self.frequency, self.rel_power,
                                              float(np.exp(key / 1000)), self.tolerance,
                                              self.fap)[0]
        return self._cache[key]
