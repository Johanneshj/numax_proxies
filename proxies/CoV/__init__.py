"""
Functions for Coefficients of Variation method
"""

from .calculate_coefficients import (
    calculate_CoV,
    bin_spectrum,
    smooth_CoV_values,
    numax_estimate_CoV,
)
from .plot_CoV import plot_CoV_vs_bin_centers, plot_supNyq_spec

__all__ = [
    "calculate_CoV",
    "bin_spectrum",
    "smooth_CoV_values",
    "numax_estimate_CoV",
    "plot_CoV_vs_bin_centers",
    "plot_supNyq_spec",
]
