import numpy as np
import lightkurve as lk
import time as t

def mean_psd(lc=None, cadence=None) :
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

    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value

    dt = np.median(np.diff(time))
    print(dt, 0.5 * (30/(30*2*24)))
    len_chunk = 90
        
    size_chunk = int(len_chunk / dt)
    n_chunk = len(time) // size_chunk
   
    # Removing the final points of the time series
    # to have the length being a multiple of n_chunk
    # and reshaping

    time     = time[:size_chunk*n_chunk].reshape((n_chunk, size_chunk))
    flux     = flux[:size_chunk*n_chunk].reshape((n_chunk, size_chunk))
    flux_err = flux_err[:size_chunk*n_chunk].reshape((n_chunk, size_chunk))

    # Initialising with first chunk
    start = t.time()    
    freq, psd = calculate_psd_for_mean_psd(time[0], flux[0], flux_err[0], freq_grid=None)

    psd = [calculate_psd_for_mean_psd(t_chunk, f_chunk, ferr_chunk, freq_grid=freq) for t_chunk, f_chunk, ferr_chunk in zip(time, flux, flux_err)]
    psd = sum(psd)

    psd /= n_chunk

    pg = lk.periodogram.Periodogram(frequency=freq, power=psd)
    end = t.time()
    print(f"computation time for averaged psd: {end - start:.4f} seconds")

    return pg

def calculate_psd_for_mean_psd(time, flux, flux_err, freq_grid):
    '''
        Calculate PSD using Lightkurve's .to_periodogram()

        Inputs:
            lc          : lightkurve LightCurve object
            args        : only applies if lc is not provided
            kwargs      : only applies if lc is not provided
        
        Outputs:
            frequeancy  : lightkurve Periodogram object
            power_sd    : power spectral density
    '''
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    pg = lc.to_periodogram(freq_unit='uHz', normalization='psd', center_data=True, frequency=freq_grid)
    frequency = pg.frequency
    power_sd = pg.power
    if freq_grid is None:  
      return frequency, power_sd
    else:
        return power_sd