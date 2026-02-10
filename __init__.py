from .NumaxProxies import NumaxProxies
from .data_preparation import GetLightcurve, mean_psd, calculate_noise, DataProcessing, read_json_file
from .plotting import plot_spectrum_with_all_numax_estimates
from .proxies.ACF import calculate_two_dim_ACF, collapsed_acf, fit_gauss_to_collapsed_acf, calculate_relative_power
from .proxies.CoV import calculate_CoV, bin_spectrum, plot_CoV_vs_bin_centers, smooth_CoV_values, numax_estimate_CoV
from .proxies.ScalingRelations import query_gaia, return_dict, get_query

# Top level
__all__ = [
    "NumaxProxies"
]
# Data preparation
__all__ += [
    "GetLightcurve",
    "mean_psd",
    "calculate_noise",
    "DataProcessing",
    "read_json_file"
]
# ACF routines
__all__ += [
    "calculate_two_dim_ACF",
    "collapsed_acf",
    "fit_gauss_to_collapsed_acf",
    "calculate_relative_power"
]
# CoV routines
__all__ += [
    "calculate_CoV",
    "bin_spectrum",
    "plot_CoV_vs_bin_centers",
    "smooth_CoV_values",
    "numax_estimate_CoV"
]
# Gaia query routines
__all__ += [
    "query_gaia",
    "return_dict",
    "get_query"
]
# Top level plotting
__all__ += [
    "plot_spectrum_with_all_numax_estimates"
]
