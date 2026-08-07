import numpy as np
from scipy.interpolate import interp1d
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator as RGI
from numpy.typing import NDArray

def resample_FAP_values(bin_centers, database):
    """Generate a database of FAP values resampled to common set of bin centers"""
    resampled_database = {}
    for key, d in database.items():

        # Window length for Savitzky-Golay filter
        wl = len(d['bin_centers']) // 3
        if wl % 2 == 0:
            wl += 1

        # 95% threshold
        filtered = savgol_filter(d['FAP_95p0'], window_length=wl, polyorder=2)
        FAP95p0 = interp1d(
            d['bin_centers'],
            filtered,
            kind='cubic',
            bounds_error=False,
            fill_value=filtered[-1]
        )

        # 99% threshold
        filtered = savgol_filter(d['FAP_99p0'], window_length=wl, polyorder=2)
        FAP99p0 = interp1d(
            d['bin_centers'],
            filtered,
            kind='linear',
            bounds_error=False,
            fill_value=filtered[-1]
        )

        # 99.9% threshold
        filtered = savgol_filter(d['FAP_99p9'], window_length=wl, polyorder=2)
        FAP99p9 = interp1d(
            d['bin_centers'],
            filtered,
            kind='linear',
            bounds_error=False,
            fill_value=filtered[-1]
        )

        # Resampled database
        resampled_database[key] = {
            "bin_centers": bin_centers,
            "FAP_95p0": FAP95p0(bin_centers),
            "FAP_99p0": FAP99p0(bin_centers),
            "FAP_99p9": FAP99p9(bin_centers),
        }
    
    return resampled_database

def grid_interpolator(resampled_database, FAP_threshold):
    """Interpolate the resampled FAP values"""
    dims = [
        sorted({k[i] for k in resampled_database})
        for i in range(4)
    ]

    lengths, cadences, overlap_factors, windowsize_factors = dims
    N = len(next(iter(resampled_database.values()))["bin_centers"])

    values = np.full(tuple(len(d) for d in dims) + (N,), np.nan)

    for i, L in enumerate(lengths):
        for j, C in enumerate(cadences):
            for k, O in enumerate(overlap_factors):
                for l, W in enumerate(windowsize_factors):

                    key = (L, C, O, W)

                    if key in resampled_database:
                        values[i, j, k, l] = (
                            resampled_database[key][f"FAP_{FAP_threshold}"]
                        )

    interp = RGI(
        (lengths,
        cadences,
        overlap_factors,
        windowsize_factors),
        values,
        bounds_error=False,
        fill_value=np.nan
    )
    
    return interp

def evaluate_faps(
        bin_centers : list[NDArray], 
        length : float, 
        cadence : float, 
        overlap_scales : list,
        width_factors : list,
        FAP_threshold : str
    ):

    # Load FAP values based on realizations of white noise
    file = 'numax_proxies/proxies/CoV/FAP_database.pkl'
    with open(file, "rb") as f:
        database = pickle.load(f)

    # Resample FAP database to common bin centers
    resampled_databases = []
    for bcs in bin_centers:
        resampled_databases.append(resample_FAP_values(bcs, database))

    # Create Scipy ReguarGridInterpolator object
    interpolators = []
    for resampled_database in resampled_databases:
        interpolators.append(grid_interpolator(resampled_database, FAP_threshold))

    FAPs = []

    scales = [float(overlap_scales)] if isinstance(overlap_scales, (int, float)) else [float(s) for s in overlap_scales]
    factors = [float(width_factors)] if isinstance(width_factors, (int, float)) else [float(w) for w in width_factors]

    # Truncate the parameters to what is covered by FAP interpolator
    overlap_scales = np.clip(scales, 2, 8).tolist()
    width_factors = np.clip(factors, 0.5, 1.5).tolist()

    i = 0
    for ovs in overlap_scales:
        for wf in width_factors:
            FAPs.append(interpolators[i]((length, cadence, ovs, wf)))
            i += 1
    
    # Return FAP values
    return FAPs