import json
import numpy as np


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


def evaluate_faps(
    x, L, W, filepath="numax_proxies/proxies/ACF/ACF_FAP_fit_coefficients.txt", c=0.0
):
    """
    Evaluates the FAP curve y(x) = a / (1 + exp(b * (x - c))) + d
    by reading fitted surface parameters from a text file.

    Parameters:
    -----------
    x : float or np.ndarray
        Evaluation grid points (e.g. log10(BCS)).
    L : float
        Timeseries length in days.
    W : float
        Width factor.
    filepath : str
        Path to the saved surface fit text file.
    c : float
        Shift parameter (default = 0.0).

    Returns:
    --------
    np.ndarray
        Evaluated FAP curve.
    """
    with open(filepath, "r") as f:
        model_data = json.load(f)

    degrees = model_data["degrees"]
    coeffs = model_data["coefficients"]

    logL = np.log10(L)
    params = {}

    # Evaluate 2D surface polynomials for a, b, and d
    
    FAPs = []
    for i, width in enumerate(W):
        for p in ["a", "b", "d"]:
            x_arr = np.asarray(x[i])
            deg = degrees[p]
            c_vec = np.array(coeffs[p])
            A = _build_poly2d_features(logL, width, deg)
            params[p] = (A @ c_vec)[0]
        FAP = params["a"] / (1.0 + np.exp(params["b"] * (x_arr - c))) + params["d"]
        FAPs.append(FAP)

    return FAPs
