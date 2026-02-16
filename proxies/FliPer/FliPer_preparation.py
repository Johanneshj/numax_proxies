import numpy as np
from scipy.ndimage import median_filter


def estimate_noise(pg=None, Kmag=None):
    """
    Estimate noise for given spectrum.
    Bugnet et al. (2018) estimates noise based on Kepler magnitude.
    But this will not work for TESS stars.
    Therefore we assume the last 100 bins in a given spectrum to be pure noise,
    and take the median as the noise level.

    Input:
        pg :: power spectrum

    Output:
        noise :: noise level
    """
    power = pg.power.value[-100:]
    noise = np.median(power)
    return noise


def highpass_filter(pg, cutoff):
    """
    FliPer requires a power density spectrum filtered with a 20 day high pass filter,
    and also with a 80 day high pass filter

    Input:
        pg :: power spectrum
        cutoff :: cutoff in days

    Output:
        med_filter :: the filtered power
    """
    freq = pg.frequency.value
    power = pg.power.value
    ws = 1.0 / (cutoff * 24 * 3600) * 1e6
    df = np.median(np.diff(freq))
    wp = int(ws / df)
    if wp % 2 == 0:
        wp += 1
    # print(len(freq), wp)
    med_filter = median_filter(power, size=wp, mode="reflect")
    filter_pg = np.column_stack((freq, med_filter))
    return filter_pg, med_filter
