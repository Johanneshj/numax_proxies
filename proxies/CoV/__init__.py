"""
Functions for Coefficients of Variation method
"""

# from .calculate_coefficients import (
#     calculate_CoV,
#     bin_spectrum,
#     smooth_CoV_values,
#     numax_estimate_CoV,
# )
# from .plot_CoV import plot_CoV_vs_bin_centers, plot_supNyq_spec, plot_CoV_Bell
from .generate_CoV_spectrum import *
from .evaluate_FAP import *
from .fitting_CoV import *
from .plot_CoV_updated import *

__all__ = [
    "log_numax_binning",
    "calculate_CoVs",
    "calculate_CoV",
    "smooth_CoVs",
    "evaluate_faps",
    "global_fitting",
    "plot_CoV"
]

# __all__ = [
#     "calculate_CoV",
#     "bin_spectrum",
#     "smooth_CoV_values",
#     "numax_estimate_CoV",
#     "plot_CoV_vs_bin_centers",
#     "plot_supNyq_spec",
#     "plot_CoV_Bell",
# ]
