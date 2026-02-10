from .calculate_coefficients import calculate_CoV, bin_spectrum, smooth_CoV_values, numax_estimate_CoV
from .plot_CoV import plot_CoV_vs_bin_centers

__all__ = ["calculate_CoV", "bin_spectrum",
           "smooth_CoV_values", "numax_estimate_CoV",
           "plot_CoV_vs_bin_centers"]