"""NumaxProxies: one star from input (PSD or light curve) to numax estimates.

    proxy = NumaxProxies.read_yaml("star.yaml").run()
    proxy.compute_numax_from_acf()
    proxy.compute_numax_from_CoV()
    proxy.compute_numax_from_scaling_relations()   # needs Gaia and/or STAR parameters
    proxy.plotting()
    proxy.results                                  # DataFrame: label, numax, numax_err
                                                   # (incl. numax_combined: mean of the estimates)
    proxy.comb_scan_table                          # with CONFIG.comb_scan: per method

The running median and the relative power of a spectrum are computed once and shared by the
cleaning, both methods, the comb check and the comb scan.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import ufloat

from . import lightcurve as lcp
from .binning import DEFAULT_MEDIAN_FILTER_FACTOR, running_median
from .config import GlobalConfig, load_config
from .data import GaiaData, PSDData
from .io import load_psd, save_lightcurve, save_psd
from .methods import NumaxFromACF, NumaxFromCoV, numax_from_scaling_relations
from .spectrum import averaged_psd, periodogram

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
        self.artefact_report = []
        self.dnu_estimates = {}
        self.comb_scan_results = {}
        self.cov_scan_results = {}
        self._medians = {}                   # (id(power), factor) -> (power, running median)
        self._relative = {}                  # (id(power), factor) -> (power, P / median - 1)
        self._checkers = {}                  # (id(power), factor) -> (power, CombChecker)

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
        elif self.psd_input.avg_psd_file:
            self.avg_psd = load_psd(self.psd_input.avg_psd_file)
        else:
            self._spectrum_from_lightcurve()
        clean = self.config.artefact_cleaning and self.psd is not None and self._kepler_sc_checks(
            "artefact_cleaning")
        exclude = self.config.exclude_regions and self.psd is not None
        if clean or exclude:
            # one running median for the cleaning and the methods: the cleaned regions are
            # replaced by noise at this median level, so the median of the cleaned PSD is the same
            med = self.running_median(self.psd)
        if clean:
            from .artefacts import clean_detected_artefacts
            cfg = self.config
            power, self.artefact_report = clean_detected_artefacts(
                self.psd.frequency, self.psd.psd, self.rng, cfg.artefact_fap,
                cfg.artefact_search_halfwidth, cfg.artefact_max_halfwidth, med=med)
            self.psd = PSDData(self.psd.frequency, power)
        if exclude:
            from .artefacts import replace_regions
            self.psd = PSDData(self.psd.frequency, replace_regions(
                self.psd.frequency, self.psd.psd, self.config.exclude_regions, self.rng, med))
        if clean or exclude:
            self._medians[(id(self.psd.psd), float(DEFAULT_MEDIAN_FILTER_FACTOR))] = (self.psd.psd, med)
            logger.info("%s: replaced %s muHz by noise", self.star.target,
                        self.config.exclude_regions)
        return self

    def running_median(self, spectrum: PSDData, factor: float = DEFAULT_MEDIAN_FILTER_FACTOR):
        """Running median of a spectrum over factor x envelope FWHM, computed once and reused
        (cleaning, ACF, CoV, comb scan).

        Args:
            spectrum: PSDData (self.psd or self.avg_psd).
            factor: window in envelope FWHM.

        Returns:
            Running median on the spectrum's frequency grid.
        """
        key = (id(spectrum.psd), float(factor))
        if key not in self._medians or self._medians[key][0] is not spectrum.psd:
            self._medians[key] = (spectrum.psd,
                                  running_median(spectrum.frequency, spectrum.psd, factor))
        return self._medians[key][1]

    def relative_power(self, spectrum: PSDData, factor: float = DEFAULT_MEDIAN_FILTER_FACTOR):
        """P / running median - 1 of a spectrum, computed once and reused (comb check, comb scan).

        Args:
            spectrum: PSDData.
            factor: running-median window in envelope FWHM.

        Returns:
            Relative power on the spectrum's frequency grid.
        """
        key = (id(spectrum.psd), float(factor))
        if key not in self._relative or self._relative[key][0] is not spectrum.psd:
            self._relative[key] = (spectrum.psd,
                                   spectrum.psd / self.running_median(spectrum, factor) - 1.0)
        return self._relative[key][1]

    def _comb_checker(self, spectrum: PSDData, factor: float):
        """One PSPS comb checker per spectrum and median window, shared by ACF and CoV (its
        cache of tested numax values is then shared too)."""
        from .artefacts import CombChecker
        key = (id(spectrum.psd), float(factor))
        if key not in self._checkers or self._checkers[key][0] is not spectrum.psd:
            cfg = self.config
            self._checkers[key] = (spectrum.psd, CombChecker(
                spectrum.frequency, self.relative_power(spectrum, factor), cfg.comb_check,
                cfg.comb_check_tolerance, cfg.comb_check_fap, cfg.comb_check_distance))
        return self._checkers[key][1]

    def _kepler_sc_checks(self, option: str) -> bool:
        """True if the Kepler SC checks apply (STAR.cadence == "Kepler_SC"); warn otherwise."""
        if self.star.cadence == "Kepler_SC":
            return True
        logger.warning("%s: CONFIG.%s is only used for STAR.cadence 'Kepler_SC'; ignored",
                       self.star.target, option)
        return False

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

        self.psd = periodogram(lc, cfg.oversampling, cfg.width_for_wf)
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
        spec = self._spectrum_for_methods()
        method = cls(spec, cfg, self.config, self.star.target, self.config.initial_numax)
        if not cfg.replace_below_min_freq:
            method.median = self.running_median(spec, cfg.median_filter_factor)
        if self.config.background_correction:
            method.background_median = self.running_median(spec)
        if self.config.comb_check != "off" and self._kepler_sc_checks("comb_check"):
            method.comb_check_mode = self.config.comb_check
            if not cfg.replace_below_min_freq:
                method.comb_checker = self._comb_checker(spec, cfg.median_filter_factor)
        method.compute()
        if self.config.cov_scan:                         # on the method's own estimate
            self._cov_scan(method, label, cfg.plot)
        if self.config.comb_scan:
            self._comb_scan(method, label, cfg.plot)
        if cfg.plot:
            method.plot()
        if cfg.save_info:
            method.save_all_data()
        if cfg.save_results:
            method.save_numax_estimates()
        self.numax_estimates[label] = method.numax_estimate
        if self.config.cov_scan:
            name = "ACF" if label == "numax_2DACF" else "CoV"
            acc = getattr(method, "cov_scan", {}).get("accepted")
            self.numax_estimates[f"numax_CoVscan_{name}"] = (acc["numax_cov"] if acc
                                                              else ufloat(np.nan, np.nan))
        self._update_combined()
        return method

    COMBINED_FROM = ("numax_2DACF", "numax_CoV", "numax_CoVscan_ACF", "numax_CoVscan_CoV")

    def _update_combined(self):
        """numax_estimates["numax_combined"]: the mean of the available seismic estimates (ACF,
        CoV, and the CoV-scan centres; NaN ones left out). Uncertainty: the spread (standard
        deviation) of those estimates, or the formal error if there is only one."""
        vals = [self.numax_estimates[k] for k in self.COMBINED_FROM
                if k in self.numax_estimates and np.isfinite(self.numax_estimates[k].n)]
        if not vals:
            combined = ufloat(np.nan, np.nan)
        elif len(vals) == 1:
            combined = vals[0]
        else:
            x = np.array([v.n for v in vals])
            combined = ufloat(float(x.mean()), float(x.std(ddof=1)))
        self.numax_estimates.pop("numax_combined", None)    # keep it last
        self.numax_estimates["numax_combined"] = combined

    def _cov_scan(self, method, label, plot):
        """CoV scan of the method's numax (CONFIG.cov_scan); report only.

        If the scan is not significant, the grid fits are repeated with one envelope FWHM around
        the rejected numax excluded and the new estimate is scanned (cov_scan_max_refits times).
        The refits use a copy of the grid results: the method (and the comb scan after it)
        keeps its own fits.

        Args:
            method: computed NumaxFromACF or NumaxFromCoV.
            label: key in numax_estimates.
            plot: also save CovScan_<ACF|CoV>.png (the accepted scan, else the first).
        """
        import copy
        from .cov_scan import cov_scan, significant
        cfg, name = self.config, "ACF" if label == "numax_2DACF" else "CoV"
        spec = self._spectrum_for_methods()
        rel = self.relative_power(spec)
        saved = copy.deepcopy(method.results), getattr(method, "numax_override", None)
        original = method.numax_estimate.n
        rejected, attempts, accepted = [], [], None
        try:
            for _ in range(cfg.cov_scan_max_refits + 1):
                start = method.numax_estimate.n
                if not np.isfinite(start):
                    break
                r = cov_scan(spec.frequency, rel, start, cfg.cov_scan_width, cfg.cov_scan_span)
                if r is None:
                    break
                r["significant"] = significant(r, cfg.cov_scan_threshold)
                attempts.append(r)
                if r["significant"]:
                    accepted = r
                    break
                rejected.append(start)
                method.refit(rejected)
        finally:                                           # the method keeps its own fits
            method.results, method.numax_override = saved
        result = dict(original=original, rejected=rejected, attempts=attempts, accepted=accepted)
        method.cov_scan = result
        self.cov_scan_results[name] = result
        if not attempts:
            logger.warning("%s: CoV scan not possible around the %s numax", self.star.target, name)
            return
        shown = accepted or attempts[0]
        logger.info("%s: CoV scan %s: guess %.1f, shifted %.1f, centre %.1f muHz, peak z %.1f (%s%s)",
                    self.star.target, name, shown["numax"], shown["scan_centre"],
                    shown["numax_cov"].n, shown["peak_z"],
                    "significant" if accepted else "not significant",
                    f", after {len(rejected)} refit(s)" if accepted and rejected else "")
        if plot:
            from .plotting import plot_cov_scan
            plot_cov_scan(shown, rel, spec.frequency, name, cfg, self.star.target)

    def _comb_scan(self, method, label, plot):
        """Comb scan of the method's final numax (CONFIG.comb_scan).

        The scan is re-centred on its fitted centre until it converges; the converged centre
        becomes the method's numax. Without a significant envelope the estimate is rejected:
        the grid fits are repeated with one envelope FWHM around it excluded and the new
        estimate is scanned (comb_scan_max_refits times); after that the numax is NaN.

        Args:
            method: computed NumaxFromACF or NumaxFromCoV.
            label: key in numax_estimates.
            plot: also save CombScan_<ACF|CoV>.png.
        """
        from .comb_scan import recentre
        cfg, name = self.config, "ACF" if label == "numax_2DACF" else "CoV"
        spec = self._spectrum_for_methods()
        rel = self.relative_power(spec)
        original = method.numax_estimate.n
        rejected, attempts, accepted = [], [], None
        for _ in range(cfg.comb_scan_max_refits + 1):
            start = method.numax_estimate.n
            if not np.isfinite(start):
                break
            r = recentre(spec.frequency, rel, start, cfg.comb_scan_width, cfg.comb_scan_span,
                         cfg.comb_scan_threshold, cfg.comb_scan_tolerance,
                         cfg.comb_scan_max_iterations, with_map=plot)
            attempts.append(r)
            if r["status"] != "rejected":
                accepted = r
                break
            rejected.append(start)
            logger.info("%s: %s numax %.1f has no significant comb envelope; refitting without it",
                        self.star.target, name, start)
            method.refit(rejected)
        method.numax_override = accepted["numax"] if accepted else ufloat(np.nan, np.nan)
        if accepted and cfg.background_correction:      # the scan centre is uncorrected
            from .background import correct_numax
            raw = accepted["numax"]
            accepted["numax_raw"] = raw
            method.numax_override = ufloat(
                correct_numax(spec.frequency, self.running_median(spec), raw.n), raw.s)
        method.comb_scan = dict(original=original, rejected=rejected, attempts=attempts,
                                accepted=accepted, final=method.numax_override)
        self.comb_scan_results[name] = method.comb_scan
        if accepted:
            last = accepted["scans"][-1]
            logger.info("%s: comb scan %s: %.1f -> %.1f muHz (%s after %d scans, peak z %.1f)",
                        self.star.target, name, accepted["start"], accepted["numax"].n,
                        accepted["status"], len(accepted["scans"]), last["peak_z"])
        else:
            logger.warning("%s: comb scan rejected every %s numax (%s)", self.star.target, name,
                           ", ".join(f"{v:.1f}" for v in rejected) or "none")
        if plot:
            shown = accepted or next((r for r in attempts if r["scans"]), None)
            if shown and shown["scans"]:
                from .plotting import plot_comb_scan
                plot_comb_scan(shown["scans"][-1], rel, spec.frequency, name, cfg,
                               self.star.target)

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
        """Save the spectrum with all numax estimates (and PSPS.png if CONFIG.plot_psps)."""
        from .plotting import plot_spectrum_with_all_numax_estimates
        plot_spectrum_with_all_numax_estimates(self._spectrum_for_methods(), self.star,
                                               self.numax_estimates, self.config)
        if self.config.plot_psps:
            self.psps_delta_nu(plot=True)

    def psps_delta_nu(self, plot: bool = False) -> dict:
        """Delta-nu from the PSPS around the ACF and CoV numax (strongest peak near
        1/Delta-nu or 2/Delta-nu).

        Args:
            plot: also save PSPS.png.

        Returns:
            {"dnu_ACF": muHz, "dnu_CoV": muHz} for the methods that were run and have a numax.
        """
        from .artefacts import psps
        cfg, curves = self.config, []
        for label, colour, method, offset in (("ACF", "forestgreen", getattr(self, "acf", None), 0.0),
                                              ("CoV", "mediumorchid", getattr(self, "cov", None), 1.0)):
            if method is None:
                continue
            numax = method.numax_estimate.n
            r = psps(method.frequency, method.normalized_power - offset, numax,
                     cfg.comb_check_tolerance, oversampling=cfg.psps_oversampling)
            if r is None:
                continue
            self.dnu_estimates[f"dnu_{label}"] = r["spacing"]
            curves.append((label, colour, numax, r))
        if plot and curves:
            from .plotting import plot_psps
            plot_psps(curves, cfg, self.star.target)
        return self.dnu_estimates

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
            if self.comb_scan_results:
                self.comb_scan_table.to_csv(self.out_dir / f"{self.star.target}_comb_scan.txt",
                                            index=False)
            if self.cov_scan_results:
                self.cov_scan_table.to_csv(self.out_dir / f"{self.star.target}_cov_scan.txt",
                                           index=False)
        return df

    @property
    def cov_scan_table(self) -> pd.DataFrame:
        """CoV scan per method (CONFIG.cov_scan).

        Returns:
            DataFrame: method, numax_original (the method's estimate), rejected (estimates without
            a significant excess), start (estimate of the accepted scan), shifted (power-excess
            peak), numax_cov (fitted centre), numax_cov_err, peak_z, significant, n (points per
            window); for a rejected star the values of the first scan.
        """
        rows = []
        for k, c in self.cov_scan_results.items():
            a = c["accepted"] or (c["attempts"][0] if c["attempts"] else None)
            rows.append({"method": k, "numax_original": c["original"],
                         "rejected": ";".join(f"{v:.2f}" for v in c["rejected"]),
                         "start": a["numax"] if a else np.nan,
                         "shifted": a["scan_centre"] if a else np.nan,
                         "numax_cov": a["numax_cov"].n if a else np.nan,
                         "numax_cov_err": a["numax_cov"].s if a else np.nan,
                         "peak_z": a["peak_z"] if a else np.nan,
                         "significant": c["accepted"] is not None, "n": a["n"] if a else np.nan})
        return pd.DataFrame(rows)

    @property
    def comb_scan_table(self) -> pd.DataFrame:
        """Comb scan per method (CONFIG.comb_scan).

        Returns:
            DataFrame: method, numax_original (before the comb scan), rejected (numax values
            without a significant envelope), start (numax the accepted scan started from),
            numax (final, background-corrected if CONFIG.background_correction), numax_comb
            (comb-scan centre), numax_err, status, n_scans, peak_z (last scan, integer profile;
            peak_z_half, _combined, _anti, _total: the diagnostic profiles), D [muHz].
        """
        rows = []
        for k, c in self.comb_scan_results.items():
            a = c["accepted"]
            last = a["scans"][-1] if a else None
            rows.append({"method": k, "numax_original": c["original"],
                         "rejected": ";".join(f"{v:.2f}" for v in c["rejected"]),
                         "start": a["start"] if a else np.nan,
                         "numax": c["final"].n,
                         "numax_comb": a["numax"].n if a else np.nan,
                         "numax_err": a["numax"].s if a else np.nan,
                         "status": a["status"] if a else "rejected",
                         "n_scans": len(a["scans"]) if a else 0,
                         "peak_z": last["peak_z"] if a else np.nan,
                         "peak_z_half": last["profiles"]["half"]["peak_z"] if a else np.nan,
                         "peak_z_combined": last["profiles"]["combined"]["peak_z"] if a else np.nan,
                         "peak_z_anti": last["profiles"]["anti"]["peak_z"] if a else np.nan,
                         "peak_z_total": last["profiles"]["total"]["peak_z"] if a else np.nan,
                         "D": last["D"] if a else np.nan})
        return pd.DataFrame(rows)
