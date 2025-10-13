import numpy as np
import lightkurve as lk
from collections import Counter


def get_lightcurve(*args):
    """
    Fetch a light curve either from:
    1) A target ID (e.g., 'KIC12345678' or 'TIC12345678'), with optional cadence.
    2) Time, flux, and optional flux_err arrays.
    3) No arguments — fallback default target.

    Returns:
        lightkurve.LightCurve
    """
    priority = ['KASOC', 'Kepler', 'TASOC', 'SPOC', 'TGLC', 'QLP']
    # -------------------------------
    # Case 1: No arguments -> fallback
    if len(args) == 0:
        search_results = lk.search_lightcurve('KIC12008916',
                                              mission='Kepler',
                                              cadence='long',
                                              author='Kepler',
                                              quarter=np.arange(0, 60))
        id = 'KIC12008916'
        return search_results.download_all().stitch(), id

    # -------------------------------
    # Case 2: ID string (+ optional cadence)
    str_args = [a for a in args if isinstance(a, str)]
    if len(str_args) > 0:
        id = str_args[0]
        cadence = str_args[1] if len(str_args) > 1 else 'long'
        mission = 'Kepler' if 'KIC' in id.upper() else 'TESS'

        search_results = lk.search_lightcurve(target=id, mission=mission, cadence=cadence)
        if len(search_results) == 0:
            raise ValueError(f"No light curves found for {id} ({mission}).")

        author_counts = Counter(search_results.author)
        most_common = author_counts.most_common()
        author = sorted(
            most_common,
            key=lambda x: (priority.index(x[0]) if x[0] in priority else len(priority), -x[1])
        )[0][0]

        search_results = lk.search_lightcurve(target=id, mission=mission, cadence=cadence, author=author)
        return search_results.download_all().stitch(), id

    # -------------------------------
    # Case 3: Arrays (time, flux, optional flux_err)
    array_args = [np.array(a) for a in args if isinstance(a, (list, np.ndarray))]
    if len(array_args) >= 2:
        time = array_args[0]
        flux = array_args[1]
        flux_err = array_args[2] if len(array_args) > 2 else None
        return lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

    # -------------------------------
    raise ValueError("Invalid arguments. Provide either an ID string or time+flux arrays.")

def prepare_lightcurve(lc=None, *args, **kwargs):
    if lc is None:
        lc = get_lightcurve(*args, **kwargs)

    if hasattr(lc, "stitch"):
        lc = lc.stitch().remove_nans().remove_outliers(5)
    else:
        lc = lc.remove_nans().remove_outliers(5)

    flux = (lc.flux - 1)*1e6
    time = lc.time.value - lc.time.value[0]

    gap_indices = np.concatenate((np.where(np.diff(time) > 3)[0], [len(time)-1]))
    gap = 0
    for i in range(1, len(gap_indices)):
        start = gap_indices[i-1]+1
        end = gap_indices[i]
        gap = (time[start]-time[start-1]) - 3
        time[start:end+1] -= gap
    
    lc = lk.LightCurve(time=time, flux=flux, flux_err=lc.flux_err)

    return lc

def calculate_psd(lc=None, *args, **kwargs):
    if lc is None:
        lc = get_lightcurve(*args, **kwargs)
    pg = lc.to_periodogram(freq_unit='uHz', normalization='psd')
    return pg

