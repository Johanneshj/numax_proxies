import numpy as np
from typing import Dict, Tuple

def evaluate_faps(
    cacf_results    :   Dict[Tuple[float, ...], dict],
    FAP_threshold   =   '95p0',
) -> Dict[Tuple[float, ...], dict]:
    """Evalute FAP based on white noise simulations as detailed in Jørgensen et al. 2026

    Args:
        cacf_results (Dict[Tuple[float, ...], dict]): dictionary containing cacfs
        FAP_threshold (str, optional): detection threshold ('99p0' or '99p0') Defaults to '95p0'.

    Returns:
        Dict[Tuple[float, ...], dict]: cacf_results dictionary updated with FAPs
    """
    for key, data in cacf_results.items():
        
        Ns = [2**i for i in range(1, 16)]

        if FAP_threshold == '95p0':
            FAP = [2.0246086, 1.8616644, 1.6811973, 1.536011, 1.4235336, 1.3377753, 1.2745201, 1.2286038, 1.1958502, 1.1726482, 1.1550456, 1.141837, 1.1322121, 1.1254695, 1.1216263]
        elif FAP_threshold == '99p0':
            FAP = [2.8355962, 2.4541499, 2.0761378, 1.7995098, 1.6005808, 1.4571139, 1.3558602, 1.2848162, 1.2354867, 1.2011936, 1.1754777, 1.156408, 1.1426409, 1.1330239, 1.1273699]
        else:
            # Fall back to 95th if wrong input
            FAP = [2.0246086, 1.8616644, 1.6811973, 1.536011, 1.4235336, 1.3377753, 1.2745201, 1.2286038, 1.1958502, 1.1726482, 1.1550456, 1.141837, 1.1322121, 1.1254695, 1.1216263]

        N = np.array(data["N_data_points_in_bin"], dtype=float)
        bc = np.asarray(data["bin_centers"], dtype=float)

        # Continue the trend of increasing bin sizes (actually bins get smaller towards the end of the spectrum)
        dec = np.where(np.diff(N) < 0)[0]
        if len(dec) > 0:
            idx = dec[0] + 1
            N[idx:] = N[idx - 1] * (bc[idx:] / bc[idx - 1]) ** 0.764

        data["fap"] = np.interp(np.log2(N), np.log2(Ns), FAP)

    return cacf_results
