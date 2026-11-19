"""NumaxProxies: one star from input (PSD or light curve) to numax estimates.

    proxy = NumaxProxies.read_yaml("star.yaml").run()
    proxy.compute_numax_from_acf()
    proxy.compute_numax_from_CoV()
    proxy.compute_numax_from_scaling_relations()   # needs Gaia and/or STAR parameters
    proxy.plotting()
    proxy.results                                  # DataFrame: label, numax, numax_err
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import lightcurve as lcp
from .config import GlobalConfig, load_config
from .data import GaiaData, PSDData
from .io import load_psd, save_lightcurve, save_psd
from .methods import NumaxFromACF, NumaxFromCoV, numax_from_scaling_relations
from .spectrum import averaged_psd, clean_kepler_sc_psd, periodogram

logger = logging.getLogger(__name__)


class NumaxProxies:

    def __init__(self, global_config: GlobalConfig):
        """Args:
            global_config: GlobalConfig (see load_config / config_from_dict).
        """
        self.settings = global_config
        self.star = global_config.star
        self.config = global_config.config
        self.lc_input = global_config.lightcurve
        self.psd_input = global_config.psd
        self.acf_config = global_config.acf_config
        self.cov_config = global_config.cov_config

        self.numax_estimates = {}
        self.gaia_data = GaiaData()
        self.lc = None
        self.psd = None
        self.avg_psd = None
        self.rng = np.random.default_rng(self.config.random_seed)

    @classmethod
    def read_yaml(cls, yaml_path: str) -> "NumaxProxies":
        """Create from a YAML configuration file.

        Args:
            yaml_path: path to the YAML file.

        Returns:
            NumaxProxies (call run() next).
        """
        return cls(load_config(yaml_path))

    @classmethod
    def from_psd(cls, frequency, power, settings: dict | GlobalConfig | None = None,
                 target: str = "unknown") -> "NumaxProxies":
        """Create from a spectrum in memory; no run() needed.

        Args:
            frequency: frequency grid [muHz].
            power: PSD.
            settings: dict with the YAML layout, GlobalConfig, or None (defaults).
            target: star name.

        Returns:
            NumaxProxies.
        """
        from .config import config_from_dict
        cfg = settings if isinstance(settings, GlobalConfig) else config_from_dict(settings or {})
        if cfg.star.target is None:
            cfg.star.target = target
        proxy = cls(cfg)
        proxy.psd = PSDData(np.asarray(frequency, dtype=float), np.asarray(power, dtype=float))
        return proxy

    @property
    def out_dir(self) -> Path:
        """Output folder <results_directory>/<target>."""
        return Path(self.config.results_directory) / str(self.star.target)

    # ----------------------------------------------------------------------------------
    # Input
    # ----------------------------------------------------------------------------------
    def run(self) -> "NumaxProxies":
        """Load the PSD file(s), or compute the spectrum from the light curve.

        Returns:
            self.
        """
        if self.config.query_gaia:
            from .gaia import query_gaia
            self.gaia_data = query_gaia(self.star.target)

        if self.psd_input.psd_file:
            self.psd = load_psd(self.psd_input.psd_file)
            if self._clean_kepler_sc:
                self.psd = PSDData(self.psd.frequency,
                                   clean_kepler_sc_psd(self.psd.frequency, self.psd.psd, self.rng))
        elif self.psd_input.avg_psd_file:
            self.avg_psd = load_psd(self.psd_input.avg_psd_file)
        else:
            self._spectrum_from_lightcurve()
        return self

    @property
    def _clean_kepler_sc(self) -> bool:
        """Clean Kepler SC artefacts? (clean_kepler_sc; None = if cadence is Kepler_SC)."""
        flag = self.config.clean_kepler_sc
        return self.star.cadence == "Kepler_SC" if flag is None else bool(flag)

    def _spectrum_from_lightcurve(self):
        """Load and process the light curve, then compute the PSD (and averaged PSD)."""
        cfg = self.config
        lc = lcp.get_lightcurve(self.star, self.lc_input)
        if cfg.sort:
            lc = lcp.sort_by_time(lc)
        if cfg.normalize:
            lc = lcp.normalize_to_ppm(lc)
        if cfg.close_gaps:
            lc = lcp.close_gaps(lc, cfg.gap_size_days)
        if cfg.save_lc:
            save_lightcurve(lc, self.out_dir / "data" / f"LC_{self.star.target}.parquet")
        if cfg.add_noise:
            lc = lcp.inject_noise(lc, cfg.noise_std, self.rng)
        trend = raw_flux = None
        if cfg.savgol:
            raw_flux = lc.flux
            lc, trend = lcp.savgol_detrend(lc, cfg.savgol_window)
        self.lc = lc

        self.psd = periodogram(lc, cfg.oversampling, cfg.width_for_wf, cfg.weight_psd_by_flux_err)
        if cfg.do_avg_psd:
            self.avg_psd = averaged_psd(lc, cfg.avg_psd_chunk, cfg.avg_psd_overlap,
                                        cfg.oversampling, cfg.avg_psd_method, cfg.n_threads)
        # NOTE: as before, the Kepler SC cleaning is only applied to PSD files (PSD.psd_file)

        if cfg.plot_lc:
            from .plotting import plot_lightcurve_and_psd
            plot_lightcurve_and_psd(lc, self.psd, cfg, self.star.target, trend, raw_flux)
        if cfg.save_psd:
            save_psd(self.psd, self.out_dir / f"PSD_{self.star.target}.parquet")
        if cfg.save_avgpsd and self.avg_psd is not None:
            save_psd(self.avg_psd, self.out_dir / f"AvgPSD_{self.star.target}.parquet")

    def _spectrum_for_methods(self) -> PSDData:
        """The averaged PSD if requested (or the only one available), else the PSD."""
        if self.psd is None and self.avg_psd is None:
            raise RuntimeError("No spectrum: call run() first")
        if (self.config.do_avg_psd or self.psd is None) and self.avg_psd is not None:
            return self.avg_psd
        return self.psd

    # ----------------------------------------------------------------------------------
    # Estimates
    # ----------------------------------------------------------------------------------
    def _run_method(self, cls, cfg, label):
        """Run one method, then plot / save as configured.

        Args:
            cls: NumaxFromACF or NumaxFromCoV.
            cfg: its config section.
            label: key in numax_estimates.

        Returns:
            The method object.
        """
        method = cls(self._spectrum_for_methods(), cfg, self.config, self.star.target,
                     self.config.initial_numax).compute()
        if cfg.plot:
            method.plot()
        if cfg.save_info:
            method.save_all_data()
        if cfg.save_results:
            method.save_numax_estimates()
        self.numax_estimates[label] = method.numax_estimate
        return method

    def compute_numax_from_acf(self):
        """Run the ACF method; result in numax_estimates['numax_2DACF'] and self.acf."""
        self.acf = self._run_method(NumaxFromACF, self.acf_config, "numax_2DACF")
        return self.numax_estimates["numax_2DACF"]

    def compute_numax_from_CoV(self):
        """Run the CoV method; result in numax_estimates['numax_CoV'] and self.cov."""
        self.cov = self._run_method(NumaxFromCoV, self.cov_config, "numax_CoV")
        return self.numax_estimates["numax_CoV"]

    def compute_numax_from_scaling_relations(self):
        """numax from scaling relations with Gaia (query_gaia) and/or STAR parameters."""
        estimates = numax_from_scaling_relations(self.star, self.gaia_data)
        if not estimates:
            logger.warning("%s: no stellar parameters for the scaling relations", self.star.target)
        self.numax_estimates.update(estimates)
        return estimates

    # ----------------------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------------------
    def plotting(self):
        """Save the spectrum with all numax estimates."""
        from .plotting import plot_spectrum_with_all_numax_estimates
        plot_spectrum_with_all_numax_estimates(self._spectrum_for_methods(), self.star,
                                               self.numax_estimates, self.config)

    @property
    def results(self) -> pd.DataFrame:
        """All estimates (saved as csv if CONFIG.save_results).

        Returns:
            DataFrame with columns label, numax, numax_err.
        """
        df = pd.DataFrame([{"label": k, "numax": getattr(v, "n", v),
                            "numax_err": getattr(v, "s", np.nan)}
                           for k, v in self.numax_estimates.items()])
        if self.config.save_results:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.out_dir / f"{self.star.target}_results.txt", index=False)
        return df
