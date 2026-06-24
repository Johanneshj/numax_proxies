from dataclasses import dataclass, field, fields
from typing import Optional, Literal
from numpy.typing import NDArray
from uncertainties import UFloat
import numpy as np

@dataclass
class StarInfo:
    """Gather information on star"""
    target          :   Optional[str] = None
    cadence         :   Optional[str] = None
    author          :   Optional[str] = None
    quarter         :   Optional[list] = None
    sector          :   Optional[list] = None
    mission         :   Optional[str] = None
    logg            :   Optional[list[float]] = None
    logg_err        :   Optional[list[float]] = None
    teff            :   Optional[list[float]] = None
    teff_err        :   Optional[list[float]] = None
    radius          :   Optional[list[float]] = None
    radius_err      :   Optional[list[float]] = None
    mass            :   Optional[list[float]] = None
    mass_err        :   Optional[list[float]] = None
    luminosity      :   Optional[list[float]] = None
    luminosity_err  :   Optional[list[float]] = None
    mag             :   Optional[float] = None

@dataclass
class LightCurveInput:
    """Inputs for getting light curve"""
    lc_file             :   Optional[str] = None
    fits_file_folder    :   Optional[str] = None

@dataclass
class PSDInput:
    """Inputs for getting PSDs"""
    psd_file            : Optional[str] = None
    avg_psd_file        : Optional[str] = None

@dataclass
class ProcessingConfig:
    """Dataclass containing all processing configuration options"""
    normalize           :   bool = True
    savgol              :   bool = False
    query_gaia          :   bool = False
    do_avg_psd          :   bool = False
    add_noise           :   bool = False
    sort                :   bool = False
    close_gaps          :   bool = False
    plot_lc             :   bool = False
    plot_all_estimates  :   bool = False
    save_lc             :   bool = False
    save_psd            :   bool = False
    save_avgpsd         :   bool = False
    save_results        :   bool = False

    oversampling    :   float = 1.0
    width_for_wf    :   Optional[float] = None
    noise_std       :   float = 0.0
    savgol_window   :   float = 90.0
    avg_psd_chunk   :   float = 90.0
    initial_numax   :   Optional[float] = None
    gap_size_days   :   float = 3.0

@dataclass
class ACFConfig:
    """ACF configuration"""
    plot                    :   Optional[str] = False
    sliding_window_style    :   Literal["linear", "logarithmic", "log_numax"] = "log_numax"
    min_freq                :   Optional[float] = None
    max_freq                :   Optional[float] = None
    min_num_points          :   Optional[int] = None
    overlap_scale           :   Optional[float] = None
    width_factor            :   Optional[float] = None
    max_acf_fit_iterations  :   int = 1
    n_sigma_numax_acf       :   float = 2 
    save_info               :   Optional[str] = False

@dataclass
class COVConfig:
    """CoV configuration"""
    plot                        :   Optional[str] = False
    overlap_factor              :   Optional[float] = None
    min_freq                    :   Optional[float] = None
    smoothing_width_factor      :   Optional[float] = None
    use_welch                   :   Optional[str] = False
    welch_seg_size              :   Optional[float] = None
    use_linear_bins             :   Optional[bool] = False  
    use_Bell                    :   Optional[bool] = False 
    save_info                   :   Optional[str] = False

@dataclass
class EACFConfig:
    """EACF configuration"""
    plot    :   Optional[bool] = False

@dataclass
class GlobalConfig:
    """Combines all sub-configs into a single state matching the YAML layout."""
    star: StarInfo = field(default_factory=StarInfo)
    lightcurve: LightCurveInput = field(default_factory=LightCurveInput)
    psd: PSDInput = field(default_factory=PSDInput)
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    acf_config: ACFConfig = field(default_factory=ACFConfig)
    cov_config: COVConfig = field(default_factory=COVConfig)
    eacf_config: EACFConfig = field(default_factory=EACFConfig)

@dataclass
class UnprocessedLightCurveData:
    """Data class containing unprocessed light curve data"""
    time        :   NDArray[np.float64]
    flux        :   NDArray[np.float64]
    flux_err    :   NDArray[np.float64]

@dataclass
class LightCurveData:
    """Data class containing light curve data"""
    time        :   NDArray[np.float64]
    flux        :   NDArray[np.float64]
    flux_err    :   NDArray[np.float64]

@dataclass 
class PSDData:
    """Data class containing PSD data"""
    frequency   :   NDArray[np.float64]
    psd         :   NDArray[np.float64]

@dataclass
class AvgPSDData:
    """Data class containing averaged PSD"""
    frequency   :   NDArray[np.float64]
    psd         :   NDArray[np.float64]

@dataclass
class GaiaData:
    """Data class containing results from Gaia query"""
    teff_gspspec        :   Optional[UFloat] = None
    teff_gspphot        :   Optional[UFloat] = None
    logg_gspspec        :   Optional[UFloat] = None
    logg_gspphot        :   Optional[UFloat] = None
    mass_flame          :   Optional[UFloat] = None
    lum_flame           :   Optional[UFloat] = None
    rad_flame           :   Optional[UFloat] = None
    rad_gspphot         :   Optional[UFloat] = None

    def has_data(self):
        """Function to check if values are present"""
        return any(
            getattr(self, field.name) is not None for field in fields(self)
        )