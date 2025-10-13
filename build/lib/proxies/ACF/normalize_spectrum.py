from scipy.ndimage import median_filter
from data_preparation.prepare_data import calculate_psd
import numpy as np


def calculate_relative_power(pg=None, *args, **kwargs):
    if pg is None:
        pg = calculate_psd(*args, **kwargs)

    frequency = pg.frequency.value
    power = pg.power.value

    nbins = 0
    bins = np.logspace(np.log10(min(frequency)), 
                    np.log10(max(frequency)), 
                    0,
                    endpoint=False
                    )
    indices = np.digitize(frequency, bins, right=False) 

    binned_freq = [frequency[np.where(indices==idx)[0]] for idx in np.unique(indices)] # bin the frequencies
    binned_power = [power[np.where(indices==idx)[0]] for idx in np.unique(indices)] # bin the powers

    bin_centers = [(freq_bin[-1] - freq_bin[0]) / 2 for freq_bin in binned_freq] # find bin centers
    df = np.mean(np.diff(frequency)) # find resolution
    if np.max(frequency) > 300:
        ws = 100 # muHz
    else:
        ws = 10 # muHz
    wls = np.ones_like(bin_centers, dtype=int) * (int(ws/df)) # calculate window sizes for median filter (100 muHz Viani 2019)
    # wls = [int(max(100/df, (2/3)*bin_center**(22/25)/df)) for bin_center in bin_centers] # alternative window size calculation
    wls = [wl if wl % 2 != 0 else wl + 1 for wl in wls] # make sure window sizes are odd

    filter = np.concatenate([median_filter(power_bin, wl, mode='reflect') for power_bin, wl in zip(binned_power, wls)])
    rel_power = (power - filter)/filter

    return rel_power, filter 