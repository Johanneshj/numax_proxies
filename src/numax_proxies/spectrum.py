"""Power spectra from light curves, averaged spectra, and Kepler short-cadence cleaning.

Periodogram routines adapted from Martin Nielsen (Porto summer school 2024).
Averaged PSD: Sylvain Breton (INAF-OACT) and Johannes Jorgensen (UIBK).
"""
import logging

import numpy as np
from astropy.timeseries import LombScargle
from scipy.integrate import simpson
from scipy.signal.windows import hann

from .data import LightCurveData, PSDData

logger = logging.getLogger(__name__)

DAY_TO_MUHZ = 1e6 / 86400   # 1 cycle/day in muHz


# --------------------------------------------------------------------------------------
# Periodogram
# --------------------------------------------------------------------------------------
def nyquist(time) -> float:
    """Nyquist frequency of a time series.

    Args:
        time: time stamps.

    Returns:
        Nyquist frequency [cycles per time unit].
    """
    return 1 / (2 * np.mean(np.diff(time)))


def spectral_window(time, df: float, nyq: float, oversampling: float = 1.0, width=None):
    """Spectral window around nyq / 2, normalised to 1 at its centre.

    Args:
        time: time stamps [days].
        df: frequency step [1/day].
        nyq: Nyquist frequency [1/day].
        oversampling: oversampling factor.
        width: half-width of the window [1/day]; None = 100 df.

    Returns:
        (frequency offsets from the centre, window power).
    """
    width = 100 * df if width is None else width
    freq_cen = 0.5 * nyq
    n_freq = int(oversampling * width / df)
    freq = freq_cen + (df / oversampling) * np.arange(-n_freq, n_freq, 1)
    x = 0.5 * np.sin(2 * np.pi * freq_cen * time) + 0.5 * np.cos(2 * np.pi * freq_cen * time)
    ls = LombScargle(time, x, center_data=True, fit_mean=False)
    power = ls.power(freq, method="fast", normalization="psd", assume_regular_frequency=True)
    power /= power[int(len(power) / 2)]
    return freq - freq_cen, power


def frequency_resolution(time, nyq: float, oversampling: float = 1.0, width=None) -> float:
    """Effective frequency resolution: integral of the spectral window.

    Args:
        time: time stamps [days].
        nyq: Nyquist frequency [1/day].
        oversampling: oversampling factor.
        width: half-width of the window [1/day]; None = 100 df.

    Returns:
        Resolution [1/day].
    """
    df = 1 / (np.nanmax(time) - np.nanmin(time))
    f, w = spectral_window(time, df, nyq, oversampling, width)
    return simpson(w, x=f)


def periodogram(lc: LightCurveData, oversampling: float = 1.0, width_for_wf=None,
                weighted: bool = False) -> PSDData:
    """Lomb-Scargle power spectral density of a light curve.

    Args:
        lc: LightCurveData (time [days], flux [ppm]).
        oversampling: oversampling factor.
        width_for_wf: half-width of the spectral window [1/day]; None = default.
        weighted: True = flux_err-weighted astropy "psd" normalisation;
            False = unweighted, calibrated with Parseval (ppm^2/muHz).

    Returns:
        PSDData (frequency [muHz], psd).
    """
    ls = LombScargle(t=lc.time, y=lc.flux, dy=lc.flux_err if weighted else None,
                     fit_mean=False, center_data=True)
    nyq = nyquist(lc.time)
    df = frequency_resolution(lc.time, nyq, oversampling, width_for_wf)
    freq = np.arange(df / oversampling, nyq, df / oversampling)
    power = ls.power(freq, normalization="psd", method="fast", assume_regular_frequency=True)
    freq = freq * DAY_TO_MUHZ
    if not weighted:
        power = power * np.var(lc.flux) / (np.sum(power) * (freq[1] - freq[0]))
    return PSDData(frequency=freq, psd=power)


# --------------------------------------------------------------------------------------
# Averaged PSD
# --------------------------------------------------------------------------------------
def averaged_psd(lc: LightCurveData, chunk_days: float = 90.0, overlap: float = 0.0,
                 oversampling: float = 1.0, method: str = "median",
                 n_threads: int | None = None) -> PSDData:
    """Average of Hann-windowed Lomb-Scargle PSDs of light-curve chunks.

    Args:
        lc: LightCurveData.
        chunk_days: chunk length [days]; shortened until < half the light curve.
        overlap: fraction of a chunk shared with the next one, in [0, 1).
        oversampling: oversampling factor.
        method: "median" or "mean".
        n_threads: threads for nifty_ls.

    Returns:
        PSDData (frequency [muHz], psd). Not chi^2_2 distributed.
    """
    import nifty_ls
    from numpy.lib.stride_tricks import sliding_window_view

    if method not in ("median", "mean"):
        raise ValueError("method must be 'median' or 'mean'")

    time = lc.time.astype(np.float64)
    flux = lc.flux.astype(np.float64)
    flux_err = lc.flux_err.astype(np.float64)

    dt = np.median(np.diff(time))
    duration = time[-1] - time[0]
    while chunk_days >= duration / 2:
        chunk_days -= 1
    size_chunk = int(chunk_days / dt)
    step = size_chunk - int(overlap * size_chunk)

    time = sliding_window_view(time * 86400, size_chunk)[::step]
    flux = sliding_window_view(flux, size_chunk)[::step]
    flux_err = sliding_window_view(flux_err, size_chunk)[::step]
    window = hann(size_chunk, sym=True)

    # common frequency grid in Hz from the chunk length
    dt = np.median(np.diff(time))
    df = 1 / (time[0, -1] - time[0, 0])
    freq_grid = np.arange(df / oversampling, 1 / (2 * dt), df / oversampling, dtype=np.float64)

    n_chunks = time.shape[0]
    res = nifty_ls.lombscargle_heterobatch(
        t_list=list(time),
        y_list=list(flux * window),
        dy_list=list(flux_err),
        normalization="psd",
        fmin_list=np.full(n_chunks, freq_grid[0], dtype=np.float64),
        fmax_list=np.full(n_chunks, freq_grid[-1], dtype=np.float64),
        Nf_list=np.full(n_chunks, len(freq_grid), dtype=int),
        nthreads=n_threads or 1,
    )
    stack = res.power_list
    psd = np.median(stack, axis=0) if method == "median" else np.mean(stack, axis=0)
    logger.info("Averaged PSD from %d chunks of %.1f d (%s)", n_chunks, chunk_days, method)
    return PSDData(frequency=freq_grid * 1e6, psd=psd)


# --------------------------------------------------------------------------------------
# Kepler short-cadence cleaning
# --------------------------------------------------------------------------------------
# Spurious Kepler short-cadence frequencies (Van Cleve et al. 2016, table 5) and the
# half-width of the replaced region as a fraction of the peak frequency.
KEPLER_SC_SPURIOUS = {
    86.8: 0.1,
    370.4: 0.05,
    2092.6: 0.05,
    2265.6: 0.05,
    2832.0: 0.05,
    4428.4: 0.05,
    4451.4: 0.05,
    4531.1: 0.05,
    5016.2: 0.02,
}


def clean_kepler_sc_psd(frequency, power, rng: np.random.Generator,
                        peaks: dict = KEPLER_SC_SPURIOUS):
    """Replace spurious Kepler short-cadence peaks with chi^2_2 noise.

    Args:
        frequency: frequency grid [muHz].
        power: PSD.
        rng: numpy random generator.
        peaks: {peak frequency [muHz]: half-width as a fraction of the peak frequency}.

    Returns:
        Copy of power; the noise mean is the mean power next to each replaced region.
    """
    power = np.array(power, dtype=float, copy=True)
    for peak, width in peaks.items():
        tol = width * peak
        mask = np.abs(frequency - peak) <= tol
        if np.any(mask):
            surrounding = (frequency > peak - 2 * tol) & (frequency < peak + 2 * tol) & ~mask
            power[mask] = rng.exponential(scale=np.mean(power[surrounding]), size=mask.sum())
    return power
