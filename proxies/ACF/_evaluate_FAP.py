import numpy as np
from scipy.interpolate import interp1d
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator as RGI
from numpy.typing import NDArray
from pathlib import Path

def resample_FAP_values(bin_centers, database, FAP_threshold):
    """Generate a database of FAP values resampled to common set of bin centers"""
    resampled_database = {}
    for key, d in database.items():
        # Window length for Savitzky-Golay filter
        wl = len(d['bin_centers']) // 3
        if wl % 2 == 0:
            wl += 1

        # 95% threshold
        filtered = savgol_filter(d[f'FAP_{FAP_threshold}'], window_length=wl, polyorder=1)
        FAP95p0 = interp1d(
            d['bin_centers'],
            filtered,
            kind='linear',
            bounds_error=False,
            fill_value=filtered[-1]
        )

        # Resampled database
        resampled_database[key] = {
            "bin_centers": bin_centers,
            f"FAP_{FAP_threshold}": FAP95p0(bin_centers)#,
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
        FAP_threshold : str,
        smoothing_factors : list
    ):

    # Load FAP values based on realizations of white noise
    PACKAGE_DIR = Path(__file__).resolve().parent
    file = PACKAGE_DIR / "FAP_database.pkl"
    with open(file, "rb") as f:
        database = pickle.load(f)
    # Resample FAP database to common bin centers
    resampled_databases = []

    # With how bin_centers is structured we need to skip every len(smoothing_factors)'th index
    for bcs in bin_centers[::len(smoothing_factors)]:
        resampled_databases.append(resample_FAP_values(bcs, database, FAP_threshold))

    # Create Scipy ReguarGridInterpolator object
    interpolators = []
    for resampled_database in resampled_databases:
        interpolators.append(grid_interpolator(resampled_database, FAP_threshold))
    FAPs = []

    scales = [float(overlap_scales)] if isinstance(overlap_scales, (int, float)) else [float(s) for s in overlap_scales]
    factors = [float(width_factors)] if isinstance(width_factors, (int, float)) else [float(w) for w in width_factors]

    # Truncate the parameters to what is covered by FAP interpolator
    overlap_scales = np.clip(scales, 2, 10).tolist()
    width_factors = np.clip(factors, 0.5, 1.5).tolist()

    i = 0
    for ovs in overlap_scales:
        for wf in width_factors:
            for sf in smoothing_factors:
                FAPs.append(interpolators[i]((length, cadence, ovs, wf)))
            i += 1
    
    # Return FAP values
    return FAPs