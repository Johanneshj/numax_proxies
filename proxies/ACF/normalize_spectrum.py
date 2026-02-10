from scipy.ndimage import median_filter
from data_preparation.prepare_data import calculate_psd
import numpy as np
import lightkurve as lk
import astropy.units as u

def calculate_relative_power(pg=None, *args, **kwargs):
    if pg is None:
        pg = calculate_psd(*args, **kwargs)

    frequency = pg.frequency.value
    power = pg.power.value

    if np.max(frequency) > 300:
        ws = 100 # muHz
    else:
        ws = 10 # muHz

    df = np.median(np.diff(frequency))
    wp = int(ws/df)

    med_filter = median_filter(power, size=wp, mode="reflect")
    rel_power = (power - med_filter)/med_filter

    normalized_pg = lk.periodogram.Periodogram(frequency=frequency*u.uHz, power=rel_power*(1/u.uHz))

    return normalized_pg, med_filter 

def powers_of_each_bin(bin_edge, frequency, power, ws):
    
    bin_indices = np.where((frequency >= (bin_edge - ws)) & (frequency < bin_edge))[0]
    powers = power[bin_indices]
    
    return powers