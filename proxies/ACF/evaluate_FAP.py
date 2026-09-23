import json
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray

def evaluate_faps(
    cacf_results: Dict[Tuple[float, ...], dict],
    L: float,
    filepath: str = "numax_proxies/proxies/ACF/ACF_FAP_fit_coefficients.txt",
    c: float = 0.0,
) -> Dict[Tuple[float, ...], dict]:
    """
    Evaluates the FAP curve y(x) = a / (1 + exp(b * (x - c))) + d
    by reading fitted surface parameters from a JSON/text file, and updates 
    each sub-dictionary in cacf_results with a 'fap' entry.

    Parameters:
    -----------
    cacf_results : dict
        Dictionary keyed by (overlap_scale, width_factor, smoothing_factor) 
        or (overlap_scale, width_factor), where each entry contains 'bin_centers'.
    L : float
        Timeseries length in days.
    filepath : str
        Path to the saved surface fit text/JSON file.
    c : float
        Shift parameter (default = 0.0).
    use_log10_bcs : bool
        If True, evaluates grid points x as log10(bin_centers) (default = True).

    Returns:
    --------
    Dict[Tuple[float, ...], dict]
        The updated cacf_results dictionary containing the 'fap' curve in each entry.
    """
    with open(filepath, "r") as f:
        model_data = json.load(f)

    degrees = model_data["degrees"]
    coeffs = model_data["coefficients"]

    logL = float(np.log10(L))

    # Cache calculated (a, b, d) parameters per width_factor to avoid duplicate matrix ops
    param_cache = {}

    for key, data in cacf_results.items():
        # Extract width_factor from key tuple: (overlap_scale, width_factor, [smoothing_factor])
        width_factor = float(key[1])

        # Evaluate 2D surface polynomial for this width_factor if not already in cache
        if width_factor not in param_cache:
            params = {}
            for p in ["a", "b", "d"]:
                deg = degrees[p]
                c_vec = np.array(coeffs[p])
                A = _build_poly2d_features(logL, width_factor, deg)
                params[p] = float((A @ c_vec)[0])
            param_cache[width_factor] = params
        else:
            params = param_cache[width_factor]

        # Prepare x evaluation points (log10 of bin centers)
        bin_centers = np.asarray(data["bin_centers"])
        x_arr = np.log10(bin_centers)

        # Evaluate logistic FAP curve
        a, b, d = params["a"], params["b"], params["d"]
        fap = a / (1.0 + np.exp(b * (x_arr - c))) + d

        # Update dictionary entry in-place
        data["fap"] = fap

    return cacf_results

def _build_poly2d_features(x, y, degree):
    """Builds 2D polynomial features for logL (x) and W (y)."""
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)

    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append((x_arr**i) * (y_arr**j))
    return np.column_stack(terms)