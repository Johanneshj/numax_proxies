"""Configuration: dataclasses matching the YAML sections, plus loading and validation.

YAML layout (all sections optional except where noted):

    STAR:        target, cadence, ..., stellar parameters for the scaling relations
    LIGHTCURVE:  lc_file / fits_file_folder
    PSD:         psd_file / avg_psd_file
    CONFIG:      global processing options
    ACF_CONFIG:  ACF method
    COV_CONFIG:  CoV method

Unknown keys raise an error (catches typos). Keys that existed in older versions but do
nothing are accepted with a warning (see DEPRECATED_KEYS).
"""
import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import Literal, Optional, Union

import numpy as np
import yaml

from .binning import DEFAULT_MEDIAN_FILTER_FACTOR

logger = logging.getLogger(__name__)


@dataclass
class StarInfo:
    """Target information. Stellar parameters must be lists (one entry per source)."""
    target: Optional[str] = None
    cadence: Optional[str] = None
    author: Optional[str] = None
    quarter: Optional[list] = None
    sector: Optional[list] = None
    mission: Optional[str] = None
    logg: Optional[list[float]] = None
    logg_err: Optional[list[float]] = None
    teff: Optional[list[float]] = None
    teff_err: Optional[list[float]] = None
    radius: Optional[list[float]] = None
    radius_err: Optional[list[float]] = None
    mass: Optional[list[float]] = None
    mass_err: Optional[list[float]] = None
    luminosity: Optional[list[float]] = None
    luminosity_err: Optional[list[float]] = None
    mag: Optional[float] = None

    def __post_init__(self):
        """Check that every stellar parameter has errors of the same length."""
        if self.target is not None:
            self.target = str(self.target)
        for name in ("logg", "teff", "radius", "mass", "luminosity"):
            val, err = getattr(self, name), getattr(self, f"{name}_err")
            if val is not None and (err is None or len(err) != len(val)):
                raise ValueError(f"STAR: '{name}' needs '{name}_err' with the same length")


@dataclass
class LightCurveInput:
    """Where to get the light curve (used only if no PSD file is given)."""
    lc_file: Optional[str] = None            # .csv, .ftr/.feather or .parquet: time [d], flux, flux_err
    fits_file_folder: Optional[str] = None   # folder with Kepler/TESS fits files


@dataclass
class PSDInput:
    """Pre-computed spectra: columns frequency [muHz], power."""
    psd_file: Optional[str] = None
    avg_psd_file: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Global processing options."""
    # light curve processing
    normalize: bool = True                  # flux -> ppm relative to the median
    sort: bool = False                      # sort by time, drop duplicates, start at t=0
    close_gaps: bool = False                # shorten gaps longer than gap_size_days
    gap_size_days: float = 3.0
    savgol: bool = False                    # subtract a Savitzky-Golay filter
    savgol_window: float = 90.0             # days
    add_noise: bool = False                 # inject white noise into the light curve
    noise_std: float = 0.0                  # ppm

    # spectra
    oversampling: float = 1.0
    width_for_wf: Optional[float] = None    # width of the spectral window (muHz-equivalent, internal units)
    do_avg_psd: bool = False                # use an averaged PSD from light-curve chunks
    avg_psd_chunk: float = 90.0             # days
    avg_psd_overlap: float = 0.0            # fraction of a chunk shared by neighbours
    avg_psd_method: Literal["median", "mean"] = "median"

    # spectrum cleaning (before the methods)
    exclude_regions: list = field(default_factory=list)   # [[lo, hi], ...] muHz replaced by noise
    artefact_cleaning: bool = False         # Kepler SC: clean listed artefacts where a peak is detected
    artefact_fap: float = 0.01              # false-alarm probability of a detection, per artefact
    artefact_search_halfwidth: float = 1.0  # muHz around each listed frequency (broad: x5)
    artefact_max_halfwidth: float = 3.0     # muHz, largest half-width replaced (broad: x5)

    # checks of the fitted numax
    comb_check: Literal["off", "near_artefacts", "always"] = "off"   # Kepler SC: PSPS comb test
                                            # of every grid fit (refit without a failed region)
    comb_check_fap: float = 0.01            # false-alarm probability of the comb test
    comb_check_tolerance: float = 0.3       # search 1/Delta-nu and 2/Delta-nu within +- 30 %
    comb_check_distance: float = 0.25       # near_artefacts: within this x envelope FWHM
    comb_scan: bool = False                 # any cadence: local 2D ACF around each final numax;
                                            # re-centre on the comb envelope (new numax) or reject
    comb_scan_width: float = 3.0            # window width [Delta-nu]
    comb_scan_span: float = 2.0             # scan numax +- span x envelope FWHM
    comb_scan_threshold: float = 5.0        # min. peak z of the envelope (white noise: ~1 %)
    comb_scan_tolerance: float = 0.1        # converged when the centre moves <= this x FWHM
    comb_scan_max_iterations: int = 5       # scans per re-centring
    comb_scan_max_refits: int = 3           # refits after rejections, then numax = NaN
    background_correction: bool = False     # numax -= alpha sigma^2 / numax (background slope)

    # Delta-nu diagnostic
    plot_psps: bool = False                 # PSPS.png: PSPS around the ACF and CoV numax
    psps_oversampling: int = 10             # zero padding of the PSPS (plots and Delta-nu only)

    # misc
    query_gaia: bool = False
    initial_numax: Optional[float] = None
    n_threads: Optional[int] = None         # Numba threads; None = Numba default (NUMBA_NUM_THREADS)
    random_seed: int = 42                   # seed for every random number used by the pipeline

    # output
    plot_lc: bool = False
    save_lc: bool = False
    save_psd: bool = False
    save_avgpsd: bool = False
    save_results: bool = False
    results_directory: str = "results"
    figure_format: Union[str, list] = "png"  # png, pdf, or [png, pdf] (figures 8.8 cm wide)

    def __post_init__(self):
        """Check the values (ranges, choices, integer counts) and normalise lists."""
        if self.avg_psd_method not in ("median", "mean"):
            raise ValueError("CONFIG.avg_psd_method must be 'median' or 'mean'")
        if not 0 <= self.avg_psd_overlap < 1:
            raise ValueError("CONFIG.avg_psd_overlap must be in [0, 1)")
        if self.comb_check not in ("off", "near_artefacts", "always"):
            raise ValueError("CONFIG.comb_check must be 'off', 'near_artefacts' or 'always'")
        for name in ("artefact_fap", "comb_check_fap"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"CONFIG.{name} must be in (0, 1)")
        if int(self.psps_oversampling) != self.psps_oversampling or self.psps_oversampling < 1:
            raise ValueError("CONFIG.psps_oversampling must be a positive integer")
        self.psps_oversampling = int(self.psps_oversampling)
        for name in ("comb_scan_width", "comb_scan_span", "comb_scan_tolerance",
                     "comb_scan_max_iterations"):
            if not getattr(self, name) > 0:
                raise ValueError(f"CONFIG.{name} must be positive")
        if int(self.comb_scan_max_refits) != self.comb_scan_max_refits or self.comb_scan_max_refits < 0:
            raise ValueError("CONFIG.comb_scan_max_refits must be a non-negative integer")
        self.comb_scan_max_refits = int(self.comb_scan_max_refits)
        self.comb_scan_max_iterations = int(self.comb_scan_max_iterations)
        if not 0 < self.comb_check_tolerance < 1:
            raise ValueError("CONFIG.comb_check_tolerance must be in (0, 1)")
        regions = self.exclude_regions or []
        if not all(len(r) == 2 and 0 <= r[0] < r[1] for r in regions):
            raise ValueError("CONFIG.exclude_regions must be [[lo, hi], ...] with 0 <= lo < hi")
        self.exclude_regions = [[float(lo), float(hi)] for lo, hi in regions]
        formats = [str(f).lower().lstrip(".") for f in np.atleast_1d(self.figure_format)]
        if not formats or any(f not in ("png", "pdf") for f in formats):
            raise ValueError("CONFIG.figure_format must be 'png', 'pdf' or a list of them")
        self.figure_format = list(dict.fromkeys(formats))


@dataclass
class _MethodConfig:
    """Options shared by the ACF and CoV methods."""
    min_freq: float = 1.0                   # muHz; first bin centre
    max_freq: Optional[float] = None        # muHz; None = end of the spectrum
    overlap_scale: list[float] = field(default_factory=lambda: [5.0])   # bins per bin width
    width_factor: list[float] = field(default_factory=lambda: [1.0])    # bin width in Delta-nu
    smoothing_factor: list[float] = field(default_factory=lambda: [1.0])  # in envelope FWHM
    median_filter_factor: float = DEFAULT_MEDIAN_FILTER_FACTOR  # running median width in FWHM
    replace_below_min_freq: bool = False    # replace PSD below min_freq with white noise
    FAP_threshold: Literal["95p0", "99p0"] = "95p0"   # per-bin white-noise level
    significance_at_numax: bool = True      # true: a significant bin must lie within 1 FWHM of numax
    plot: bool = False
    plot_log_scale: bool = False
    plot_comb_check: bool = False           # add the comb-check diagnostic (PSPS) panel
    save_info: bool = False                 # all_data.pkl.gz with every grid setting
    save_results: bool = False              # numax_estimates.txt

    def __post_init__(self):
        """Turn grid values into lists of floats and check them."""
        for name in ("overlap_scale", "width_factor", "smoothing_factor"):
            val = getattr(self, name)
            val = [float(v) for v in np.atleast_1d(val)]
            if any(v <= 0 for v in val):
                raise ValueError(f"{name} values must be positive")
            setattr(self, name, val)
        if self.min_freq is None:
            raise ValueError("min_freq must be set")
        if self.FAP_threshold not in ("95p0", "99p0"):
            raise ValueError("FAP_threshold must be '95p0' or '99p0'")


@dataclass
class ACFConfig(_MethodConfig):
    """ACF method."""
    collapse: Literal["all_lags", "delta_nu"] = "all_lags"   # how each bin's ACF becomes one number
    dnu_window: float = 0.1                 # delta_nu: half-width of each lag band (see dnu_window_mode)
    dnu_window_mode: Literal["relative", "fixed"] = "relative"   # relative: (1 +- w) m Delta-nu;
                                            # fixed: (m +- w) Delta-nu, the same width for every band
    dnu_multiples: list[float] = field(default_factory=lambda: [1.0])  # delta_nu: lags at m x Delta-nu
    dnu_bands: Optional[int] = None         # delta_nu: B -> bands at m = 0.5, 1.0, ..., 0.5 B
                                            # (overrides dnu_multiples); bands beyond a bin are skipped
    plot_2dacf: Union[bool, list] = False   # 2D ACF panel in ACF.png: true (first grid
                                            # setting) or [overlap_scale, width_factor]
    max_acf_fit_iterations: int = 1         # successive Gaussian fits; keeps the largest integral
    n_sigma_numax_acf: float = 2.0          # half-width (in fitted sigma) of the integral window

    def __post_init__(self):
        """Check collapse, the delta_nu lag window and plot_2dacf."""
        super().__post_init__()
        if self.collapse not in ("all_lags", "delta_nu"):
            raise ValueError("ACF_CONFIG.collapse must be 'all_lags' or 'delta_nu'")
        if self.dnu_bands is not None:
            if int(self.dnu_bands) != self.dnu_bands or self.dnu_bands < 1:
                raise ValueError("dnu_bands must be a positive integer")
            self.dnu_bands = int(self.dnu_bands)
            self.dnu_multiples = [0.5 * b for b in range(1, self.dnu_bands + 1)]
        self.dnu_multiples = sorted(float(m) for m in np.atleast_1d(self.dnu_multiples))
        if not 0 < self.dnu_window < 1 or any(m <= 0 for m in self.dnu_multiples):
            raise ValueError("dnu_window must be in (0, 1) and dnu_multiples positive")
        if self.dnu_window_mode not in ("relative", "fixed"):
            raise ValueError("dnu_window_mode must be 'relative' or 'fixed'")
        if self.dnu_window_mode == "fixed" and self.dnu_window >= min(self.dnu_multiples):
            raise ValueError("dnu_window_mode 'fixed': dnu_window must be < the smallest multiple")
        if self.collapse == "delta_nu":
            needed = (min(self.dnu_multiples) + self.dnu_window if self.dnu_window_mode == "fixed"
                      else (1 + self.dnu_window) * min(self.dnu_multiples))
            too_narrow = [w for w in self.width_factor if w <= needed]
            if too_narrow:
                warnings.warn(f"ACF collapse 'delta_nu': width_factor {too_narrow} <= {needed:.2f} "
                              "is narrower than the first lag band; those bins give NaN",
                              stacklevel=2)
        if not isinstance(self.plot_2dacf, bool):
            pair = [float(v) for v in np.atleast_1d(self.plot_2dacf)]
            if len(pair) != 2:
                raise ValueError("plot_2dacf must be true/false or [overlap_scale, width_factor]")
            self.plot_2dacf = pair



@dataclass
class COVConfig(_MethodConfig):
    """CoV method."""
    use_fallback: bool = True               # weighted-mean numax when no Gaussian fit is possible


@dataclass
class GlobalConfig:
    """All sections of a YAML file."""
    star: StarInfo = field(default_factory=StarInfo)
    lightcurve: LightCurveInput = field(default_factory=LightCurveInput)
    psd: PSDInput = field(default_factory=PSDInput)
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    acf_config: ACFConfig = field(default_factory=ACFConfig)
    cov_config: COVConfig = field(default_factory=COVConfig)


SECTIONS = {
    "STAR": ("star", StarInfo),
    "LIGHTCURVE": ("lightcurve", LightCurveInput),
    "PSD": ("psd", PSDInput),
    "CONFIG": ("config", ProcessingConfig),
    "ACF_CONFIG": ("acf_config", ACFConfig),
    "COV_CONFIG": ("cov_config", COVConfig),
}

# Keys of the original package (e.g. the per-star YAMLs of the earlier runs); accepted with a
# warning, they have no effect.
DEPRECATED_KEYS = {
    "CONFIG": {"plot_all_estimates"},
    "ACF_CONFIG": {"sliding_window_style"},
    "COV_CONFIG": {"use_linear_bins", "use_welch", "welch_seg_size", "use_Bell"},
}
DEPRECATED_SECTIONS = {"EACF_CONFIG"}


def _build(section: str, cls, values: dict | None):
    """Build one config section, dropping deprecated keys.

    Args:
        section: YAML section name.
        cls: dataclass of the section.
        values: the section's dictionary (or None).

    Returns:
        Instance of cls. Raises ValueError for unknown keys.
    """
    values = dict(values or {})
    for key in DEPRECATED_KEYS.get(section, set()) & values.keys():
        values.pop(key)
        warnings.warn(f"{section}.{key} has no effect and is ignored", DeprecationWarning, stacklevel=3)
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown key(s) in {section}: {sorted(unknown)}. Valid keys: {sorted(valid)}")
    return cls(**values)


def config_from_dict(d: dict) -> GlobalConfig:
    """Build and validate a configuration from a dictionary with the YAML layout.

    Args:
        d: {section name: {key: value}}.

    Returns:
        GlobalConfig.
    """
    unknown = set(d) - set(SECTIONS) - DEPRECATED_SECTIONS
    if unknown:
        raise ValueError(f"Unknown section(s): {sorted(unknown)}. Valid: {sorted(SECTIONS)}")
    for sec in DEPRECATED_SECTIONS & d.keys():
        warnings.warn(f"Section {sec} has no effect and is ignored", DeprecationWarning, stacklevel=2)
    return GlobalConfig(**{attr: _build(sec, cls, d.get(sec)) for sec, (attr, cls) in SECTIONS.items()})


def load_config(yaml_path: str) -> GlobalConfig:
    """Read and validate a YAML configuration file.

    Args:
        yaml_path: path to the YAML file. Relative input paths are taken relative to it.

    Returns:
        GlobalConfig.
    """
    from pathlib import Path

    with open(yaml_path) as f:
        cfg = config_from_dict(yaml.safe_load(f) or {})
    base = Path(yaml_path).resolve().parent
    for obj, name in ((cfg.lightcurve, "lc_file"), (cfg.lightcurve, "fits_file_folder"),
                      (cfg.psd, "psd_file"), (cfg.psd, "avg_psd_file")):
        value = getattr(obj, name)
        if value and not Path(value).is_absolute() and (base / value).exists():
            setattr(obj, name, str(base / value))
    return cfg
