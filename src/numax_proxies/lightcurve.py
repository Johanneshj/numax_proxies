"""Getting and processing light curves.

get_lightcurve() chooses the source: a light-curve file, a folder with Kepler/TESS fits
files, or a lightkurve download by target name. The processing functions each take and
return a LightCurveData object.
"""
import glob
import logging

import numpy as np
from scipy.signal import savgol_filter

from .data import LightCurveData
from .io import load_lightcurve_file

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------
def _mission_of(target: str) -> str:
    """Mission from the target name: "Kepler" (KIC...) or "TESS" (TIC...)."""
    t = str(target).upper()
    if "KIC" in t:
        return "Kepler"
    if "TIC" in t:
        return "TESS"
    raise ValueError(f"Cannot tell the mission from target name '{target}' (expected KIC... or TIC...)")


def _from_lightkurve(lc) -> LightCurveData:
    """LightCurveData from a lightkurve object (NaNs and 5-sigma outliers removed)."""
    lc = lc.remove_nans().remove_outliers(5)
    return LightCurveData(lc.time.value, lc.flux.value, lc.flux_err.value)


def lightcurve_from_fits(folder: str, target: str) -> LightCurveData:
    """Stitch the fits files of a target in a folder.

    Args:
        folder: folder with kplr<KIC>* (Kepler) or *<TIC>* (TESS) fits files.
        target: "KIC..." or "TIC...".

    Returns:
        LightCurveData.
    """
    import lightkurve as lk

    number = int("".join(filter(str.isdigit, target)))
    pattern = (f"{folder}/kplr{number:09d}*" if _mission_of(target) == "Kepler"
               else f"{folder}/*{number:016d}*")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No fits files for target {target} ({pattern})")
    lcs = [lk.read(f, quality_bitmask="default") for f in files]
    return _from_lightkurve(lk.LightCurveCollection(lcs).stitch())


def lightcurve_from_download(target: str, author=None, cadence=None, quarter=None,
                             sector=None) -> LightCurveData:
    """Download and stitch a light curve with lightkurve.

    Args:
        target: "KIC..." or "TIC...".
        author, cadence, quarter, sector: lightkurve search options (default author:
            Kepler for KIC, SPOC for TIC).

    Returns:
        LightCurveData.
    """
    import lightkurve as lk

    mission = _mission_of(target)
    if author is None:
        author = "Kepler" if mission == "Kepler" else "SPOC"
    kwargs = dict(target=target, author=author, cadence=cadence)
    kwargs["quarter" if mission == "Kepler" else "sector"] = quarter if mission == "Kepler" else sector
    search = lk.search_lightcurve(**kwargs)
    if len(search) == 0:
        raise FileNotFoundError(f"lightkurve found no light curves for {target} ({kwargs})")
    return _from_lightkurve(search.download_all(quality_bitmask="default").stitch())


def get_lightcurve(star, lc_input) -> LightCurveData:
    """Light curve from the first available source: lc_file, fits folder, download.

    Args:
        star: StarInfo.
        lc_input: LightCurveInput.

    Returns:
        LightCurveData.
    """
    if lc_input.lc_file:
        return load_lightcurve_file(lc_input.lc_file)
    if lc_input.fits_file_folder:
        return lightcurve_from_fits(lc_input.fits_file_folder, star.target)
    if star.target:
        return lightcurve_from_download(star.target, star.author, star.cadence,
                                        star.quarter, star.sector)
    raise ValueError("No input: give PSD.psd_file, LIGHTCURVE.lc_file, "
                     "LIGHTCURVE.fits_file_folder or STAR.target")


# --------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------
def sort_by_time(lc: LightCurveData) -> LightCurveData:
    """Drop duplicate time stamps, sort, and start the time axis at 0.

    Args:
        lc: LightCurveData.

    Returns:
        LightCurveData.
    """
    _, idx = np.unique(lc.time, return_index=True)
    time, flux, err = lc.time[idx], lc.flux[idx], lc.flux_err[idx]
    order = np.argsort(time)
    time = time[order]
    return LightCurveData(time - time[0], flux[order], err[order])


def normalize_to_ppm(lc: LightCurveData) -> LightCurveData:
    """Flux -> (flux / median - 1) x 1e6; errors scaled accordingly.

    Args:
        lc: LightCurveData.

    Returns:
        LightCurveData in ppm.
    """
    med = np.nanmedian(lc.flux)
    return LightCurveData(lc.time.copy(), (lc.flux / med - 1) * 1e6, lc.flux_err / med * 1e6)


def close_gaps(lc: LightCurveData, gap_size_days: float) -> LightCurveData:
    """Shorten every gap longer than gap_size_days to gap_size_days.

    Args:
        lc: LightCurveData.
        gap_size_days: maximum gap [days].

    Returns:
        LightCurveData.
    """
    time = lc.time.copy()
    gaps = np.concatenate((np.where(np.diff(time) > gap_size_days)[0], [len(time) - 1]))
    for i in range(1, len(gaps)):
        start, end = gaps[i - 1] + 1, gaps[i]
        shift = (time[start] - time[start - 1]) - gap_size_days
        time[start:end + 1] -= shift
    return LightCurveData(time, lc.flux.copy(), lc.flux_err.copy())


def savgol_detrend(lc: LightCurveData, window_days: float, polyorder: int = 3):
    """Subtract a Savitzky-Golay filter.

    Args:
        lc: LightCurveData.
        window_days: filter window [days].
        polyorder: polynomial order.

    Returns:
        (detrended LightCurveData, filter).
    """
    dt = np.median(np.diff(lc.time))
    wl = int(window_days / dt)
    if wl % 2 == 0:
        wl += 1
    trend = savgol_filter(lc.flux, window_length=wl, polyorder=polyorder)
    return LightCurveData(lc.time.copy(), lc.flux - trend, lc.flux_err.copy()), trend


def inject_noise(lc: LightCurveData, noise_std: float, rng: np.random.Generator) -> LightCurveData:
    """Add Gaussian white noise.

    Args:
        lc: LightCurveData.
        noise_std: standard deviation [flux units, ppm].
        rng: numpy random generator.

    Returns:
        LightCurveData.
    """
    return LightCurveData(lc.time.copy(), lc.flux + rng.normal(0, noise_std, len(lc.flux)),
                          lc.flux_err.copy())
