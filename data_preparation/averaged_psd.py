import numpy as np
import lightkurve as lk
import time as t
from .dataclasses import LightCurveData
from numpy.typing import NDArray
from .data_processing import DataProcessing
# This one is techically not needed !!!!!

def mean_psd(lc : LightCurveData):
    """
    Author: Sylvain Breton
    email: sylvain.breton@inaf.it
    Created: 22 Nov 2024
    INAF-OACT

    Compute mean PSD of a light curve, by subdividing it
    into chunks of equal length. The light curve sampling
    is assumed to be regular.

    Parameters
    ----------
    param time: ndarray
      Input time vector in days

    param flux: ndarray
      Input flux of the light curve in ppm

    param len_chunk: float
      Length of the chunks in days.
      Optional, default 90

    Returns
    -------
    tuple of arrays
      A tuple with frequency and power spectral density
      vectors.
    """

    time = lc.time
    flux = lc.flux
    flux_err = lc.flux_err

    dt = np.median(np.diff(time))
    len_chunk = 90

    size_chunk = int(len_chunk / dt)
    n_chunk = len(time) // size_chunk

    # Removing the final points of the time series
    # to have the length being a multiple of n_chunk
    # and reshaping

    time = time[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))
    flux = flux[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))
    flux_err = flux_err[: size_chunk * n_chunk].reshape((n_chunk, size_chunk))

    # Initialising with first chunk
    freq, psd = calculate_psd_for_mean_psd(
        time[0], flux[0], flux_err[0], freq_grid=None
    )

    psd = [
        calculate_psd_for_mean_psd(t_chunk, f_chunk, ferr_chunk, freq_grid=freq)
        for t_chunk, f_chunk, ferr_chunk in zip(time, flux, flux_err)
    ]
    psd = sum(psd)

    psd /= n_chunk

    pg = lk.periodogram.Periodogram(frequency=freq, power=psd)

    return pg


def calculate_psd_for_mean_psd(time : NDArray, flux : NDArray, flux_err : NDArray, freq_grid : NDArray):
    """
    Calculate PSD using Lightkurve's .to_periodogram()

    Inputs:
        lc          : lightkurve LightCurve object
        args        : only applies if lc is not provided
        kwargs      : only applies if lc is not provided

    Outputs:
        frequeancy  : lightkurve Periodogram object
        power_sd    : power spectral density
    """
    dp = DataProcessing(
        time = time,
        flux = flux,
        flux_err = flux_err
    )
    dp.microHz_periodogram()
    _, psd = dp.final_psd
    
    if freq_grid is None:
        return freq_grid, psd
    else:
        return psd
