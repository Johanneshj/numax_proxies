"""Kepler short-cadence artefacts: detection-based cleaning and a comb check of fitted numax.

Both are used only for STAR.cadence == "Kepler_SC" (switches in CONFIG):
  artefact_cleaning : at each listed artefact frequency, clean the spectrum only if a
                      significant narrow peak is found there (the oscillation comb survives).
  comb_check        : after a fit, require a comb with spacing ~Delta-nu around the fitted
                      numax, tested on the autocorrelation of that stretch of spectrum
                      (Wiener-Khinchin): the lags near the spacing must be more correlated than
                      the lags on either side; "near_artefacts" (only fits near a listed
                      artefact) or "always".
The power spectrum of the power spectrum (psps) is kept as a Delta-nu diagnostic.
"""
import logging

import numpy as np
from scipy.stats import norm

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
    """Power spectrum of the power spectrum (PSPS) around numax: a Delta-nu diagnostic.

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


COMB_BANDS = ((0.90, 1.10, 1.0), (0.65, 0.85, -0.5), (1.15, 1.35, -0.5))   # (lo, hi, weight) x D
COMB_STEP = 0.05                                  # trial spacings D in steps of 5 % of Delta-nu


def acf_comb_details(frequency, rel_power, numax, tolerance=0.3, fap=0.01, n_fwhm=1.0):
    """Is there a comb around numax? Wiener-Khinchin test on the stretch of spectrum.

    The relative power x = P / median - 1 within +- n_fwhm envelope FWHM of numax is centred
    and its autocorrelation r(k) = ACF(k) / ACF(0) computed (zero-padded FFT). For a trial
    spacing D the contrast

        C(D) = mean r over [0.9, 1.1] D  -  (mean r over [0.65, 0.85] D + mean r over
               [1.15, 1.35] D) / 2

    is large for a comb (modes coincide at a shift of D, fall in the gaps at 0.75 D and
    1.25 D) and ~0 for a single spike or for broad, smooth excess (which raises all three
    bands alike). C = sum_k a_k r(k) is linear in r, so its white-noise mean and variance are
    analytic (E r = -(n-k)/n^2, Var r = (n-k)/n^2, lags conditioned on sum r = -1/2). D is
    scanned over (1 +- tolerance) Delta-nu(numax) in steps of COMB_STEP; the largest
    standardised contrast is compared with the Gaussian threshold for the maximum of that many
    trials (Sidak, conservative for these positively correlated trials).

    Args:
        frequency: frequency grid [muHz] (uniform).
        rel_power: P / median - 1 on that grid.
        numax: fitted numax [muHz].
        tolerance: relative range of trial spacings around Delta-nu(numax).
        fap: false-alarm probability of the test.
        n_fwhm: half-width of the stretch of spectrum, in envelope FWHM.

    Returns:
        dict with lag [muHz], r, spacing (best trial D [muHz]), z (its standardised contrast),
        threshold, comb (bool), bands ([(lo, hi, mean r)] at D, in muHz), dnu (Delta-nu(numax));
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
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:n]
    if not corr[0] > 0:
        return None
    r = corr / corr[0]
    d = (n - np.arange(n)) / n ** 2                       # white-noise Var r(k) (= -E r(k))
    df = frequency[1] - frequency[0]
    dnu = float(delta_nu(numax))
    trials = dnu * (1 + np.arange(-tolerance, tolerance + 1e-9, COMB_STEP))
    best_z, best_d, n_trials = -np.inf, np.nan, 0
    for spacing in trials:
        a = np.zeros(n)
        for lo, hi, w in COMB_BANDS:
            k = np.arange(int(np.ceil(lo * spacing / df)), int(np.floor(hi * spacing / df)) + 1)
            k = k[(k >= 1) & (k < n)]
            if k.size == 0:
                break
            a[k] += w / k.size
        else:
            mu = -np.sum(a * d)
            var = np.sum(a * a * d) - mu ** 2 * 2 * n / (n - 1)
            if var <= 0:
                continue
            z = (np.sum(a * r) - mu) / np.sqrt(var)
            n_trials += 1
            if z > best_z:
                best_z, best_d = z, spacing
    if n_trials == 0:
        return None
    threshold = norm.isf(-np.expm1(np.log1p(-fap) / n_trials))
    lag = np.arange(n) * df
    bands = []
    for lo, hi, _ in COMB_BANDS:
        m = (lag >= lo * best_d) & (lag <= hi * best_d) & (np.arange(n) >= 1)
        bands.append((lo * best_d, hi * best_d, float(np.mean(r[m])) if m.any() else np.nan))
    return dict(lag=lag, r=r, spacing=float(best_d), z=float(best_z), threshold=float(threshold),
                comb=bool(best_z > threshold), bands=bands, dnu=dnu)


def acf_comb_test(frequency, rel_power, numax, tolerance=0.3, fap=0.01, n_fwhm=1.0):
    """Is there a comb around numax? (see acf_comb_details)

    Returns:
        (comb present, largest standardised contrast, its spacing D [muHz]).
    """
    d = acf_comb_details(frequency, rel_power, numax, tolerance, fap, n_fwhm)
    if d is None:
        return False, np.nan, np.nan
    return d["comb"], d["z"], d["spacing"]


class CombChecker:
    """numax -> passes the comb check? (results cached on numax in 0.1 % steps).

    Args:
        frequency: frequency grid [muHz].
        rel_power: P / median - 1.
        mode: "near_artefacts" (test only fits within distance x envelope FWHM of a listed
            artefact) or "always".
        tolerance, fap: see acf_comb_test.
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
            self._cache[key] = acf_comb_test(self.frequency, self.rel_power,
                                             float(np.exp(key / 1000)), self.tolerance,
                                             self.fap)[0]
        return self._cache[key]
