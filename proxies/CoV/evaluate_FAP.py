import numpy as np
from typing import Dict, Tuple

def evaluate_faps(
    CoV_results: Dict[Tuple[float, ...], dict],
    FAP_threshold = '95p0',
) -> Dict[Tuple[float, ...], dict]:

    for key, data in CoV_results.items():
        
        Ns = [2**i for i in range(1, 17)]

        if FAP_threshold == '95p0':
            FAP = [1.208707, 1.281996, 1.285703, 1.221684, 1.181587, 1.135029, 1.099971, 1.071177, 1.051736, 1.036010, 1.025891, 1.018225, 1.011941, 1.008948, 1.006825, 1.004831]
        elif FAP_threshold == '99p0':
            FAP = [1.425548, 1.540136, 1.512063, 1.406365, 1.297513, 1.212219, 1.151909, 1.109370, 1.075605, 1.051375, 1.037503, 1.026631, 1.017591, 1.012366, 1.008631, 1.006162]
        else:
            # Fall back to 95th if wrong input
            FAP = [1.208707, 1.281996, 1.285703, 1.221684, 1.181587, 1.135029, 1.099971, 1.071177, 1.051736, 1.036010, 1.025891, 1.018225, 1.011941, 1.008948, 1.006825, 1.004831]

        N = np.array(data["N_data_points_in_bin"], dtype=float)
        bc = np.asarray(data["bin_centers"], dtype=float)

        # Continue the trend of increasing bin sizes (actually bins get smaller towards the end of the spectrum)
        dec = np.where(np.diff(N) < 0)[0]
        if len(dec) > 0:
            idx = dec[0] + 1
            N[idx:] = N[idx - 1] * (bc[idx:] / bc[idx - 1]) ** 0.764

        data["fap"] = np.interp(np.log2(N), np.log2(Ns), FAP)

    return CoV_results
