"""NumaxProxies: one star from input (PSD or light curve) to numax estimates.

    proxy = NumaxProxies.read_yaml("star.yaml").run()
    proxy.compute_numax_from_acf()
    proxy.compute_numax_from_CoV()
    proxy.compute_numax_from_acv()                 # optional third proxy
    proxy.compute_numax_from_scaling_relations()   # needs Gaia and/or STAR parameters
    proxy.plotting()
    proxy.results                                  # label, numax, numax_err (incl. numax_combined)
    proxy.comb_scan_table, proxy.cov_scan_table    # with CONFIG.comb_scan / cov_scan
    proxy.entropy_check_table                      # with CONFIG.entropy_check

The running median and relative power of a spectrum are computed once and shared.
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
from .methods import NumaxFromACF, NumaxFromACV, NumaxFromCoV, numax_from_scaling_relations
from .spectrum import averaged_psd, periodogram

logger = logging.getLogger(__name__)


class NumaxProxies:

    def __init__(self, global_config: GlobalConfig):
        """Set up from a GlobalConfig (see load_config / config_from_dict)."""
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
        self.entropy_check_results = {}
        self._medians = {}                   # (id(power), factor) -> (power, running median)
        self._relative = {}                  # (id(power), factor) -> (power, P / median - 1)
        self._checkers = {}                  # (id(power), factor) -> (power, CombChecker)

    @classmethod
    def read_yaml(cls, yaml_path: str) -> "NumaxProxies":
        """Create from a YAML configuration file (call run() next)."""
        return cls(load_config(yaml_path))

    @classmethod
    def from_psd(cls, frequency, power, settings: dict | GlobalConfig | None = None,
                 target: str = "unknown") -> "NumaxProxies":
        """Create from a spectrum in memory (no run() needed); settings use the YAML layout."""
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
        """Load the PSD file(s), or compute the spectrum from the light curve."""
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
        """Running median of a spectrum over factor x envelope FWHM (computed once, cached)."""
        key = (id(spectrum.psd), float(factor))
        if key not in self._medians or self._medians[key][0] is not spectrum.psd:
            self._medians[key] = (spectrum.psd,
                                  running_median(spectrum.frequency, spectrum.psd, factor))
        return self._medians[key][1]

    def relative_power(self, spectrum: PSDData, factor: float = DEFAULT_MEDIAN_FILTER_FACTOR):
        """P / running median - 1 of a spectrum (computed once, cached)."""
        key = (id(spectrum.psd), float(factor))
        if key not in self._relative or self._relative[key][0] is not spectrum.psd:
            self._relative[key] = (spectrum.psd,
                                   spectrum.psd / self.running_median(spectrum, factor) - 1.0)
        return self._relative[key][1]

    def _comb_checker(self, spectrum: PSDData, factor: float):
        """One PSPS comb checker per spectrum and median window, shared by the methods."""
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
        """Run one method with the configured checks (CoV scan, comb scan), then plot and save."""
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
        if self.config.entropy_check:                    # also on the method's own estimate
            self._entropy_check(method, cfg.plot)
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
            name = method.name
            acc = getattr(method, "cov_scan", {}).get("accepted")
            self.numax_estimates[f"numax_CoVscan_{name}"] = (acc["numax_cov"] if acc
                                                              else ufloat(np.nan, np.nan))
        if self.config.comb_scan:
            self.numax_estimates[f"numax_ACVscan_{method.name}"] = getattr(
                method, "comb_scan", {}).get("acv_numax", ufloat(np.nan, np.nan))
        self._update_combined()
        return method

    COMBINED_FROM = ("numax_2DACF", "numax_CoV", "numax_CoVscan_ACF", "numax_CoVscan_CoV")

    def _update_combined(self):
        """numax_combined: mean of the ACF, CoV and CoV-scan estimates; error = their spread."""
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
        """CoV scan of the method's numax (report only).

        If not significant, the scan is repeated on refitted estimates (cov_scan_max_refits); the
        refits use a copy, so the method keeps its own fits.
        """
        import copy
        from .cov_scan import cov_scan, significant
        cfg, name = self.config, method.name
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

    def _entropy_check(self, method, plot):
        """Entropy check of the method's numax for every (width, bins, peaks, noise cut) setting, at
        the initial and the shifted numax (report only).

        numax_H<version>_<method> / dnu_H<version>_<method>: median of the reported numax / D over
        the significant settings (error: their spread).
        """
        from .entropy_check import VERSIONS, entropy_sweep, window_widths
        cfg, name = self.config, method.name
        spec = self._spectrum_for_methods()
        rel = self.relative_power(spec)
        numax = method.numax_estimate.n
        widths = window_widths(numax if np.isfinite(numax) else 0.0, cfg.entropy_check_boundary,
                               cfg.entropy_check_width_power, cfg.entropy_check_width_peaks)
        sw = entropy_sweep(spec.frequency, rel, numax, widths, cfg.entropy_check_bins,
                           cfg.entropy_check_peaks, cfg.entropy_check_peak_gate,
                           cfg.entropy_check_fap)
        method.entropy_check = sw
        self.entropy_check_results[name] = sw
        summary = []
        for v in VERSIONS:
            rows = [r for r in sw["rows"] if r["version"] == v]
            ok = [r for r in rows if np.isfinite(r["numax"])]
            nu = np.array([r["numax"] for r in ok])
            self.numax_estimates[f"numax_H{v}_{name}"] = (
                ufloat(float(np.median(nu)), float(nu.std(ddof=1)) if nu.size > 1 else np.nan)
                if ok else ufloat(np.nan, np.nan))
            if ok:
                self.dnu_estimates[f"dnu_H{v}_{name}"] = float(np.median([r["D"] for r in ok]))
            summary.append(f"{v} {len(ok)}/{len(rows)} settings"
                           + (f", numax {np.median(nu):.1f}" if ok else ""))
        logger.info("%s: entropy check %s (guess %.1f muHz): %s", self.star.target, name,
                    method.numax_estimate.n, "; ".join(summary))
        if plot:
            shown = self._entropy_shown(sw, cfg)
            if shown:
                from .plotting import plot_entropy_check
                plot_entropy_check(shown, name, cfg, self.star.target)

    @staticmethod
    def _entropy_shown(sw, cfg):
        """The checks to plot, {version: check}: the middle bin number, the fewest peaks and the
        first noise cut; the initial window if either version reports it, else the shifted one."""
        from .entropy_check import VERSIONS, peak_numbers
        bins = sorted(cfg.entropy_check_bins)
        nb = bins[len(bins) // 2]
        n = peak_numbers(sw["widths"]["peaks"], cfg.entropy_check_peaks)[0]
        g = cfg.entropy_check_peak_gate[0]
        cut = "none" if g is None else g
        rows = [r for r in sw["rows"] if r["bins"] == nb
                and (r["version"] == "power" or (r["peaks"] == n and r["noise_cut"] == cut))]
        aim = ("none" if any(r["reported"] == "initial" for r in rows)
               or not any(r["reported"] == "shifted" for r in rows) else "excess")
        keys = dict(power=(nb, None, None), peaks=(nb, n, g))
        shown = {}
        for v in VERSIONS:
            c = sw["checks"][(v, aim, *keys[v])] or sw["checks"][(v, "none", *keys[v])]
            if c is not None:
                shown[v] = c
        return shown

    def _comb_scan(self, method, label, plot):
        """Comb scan of the method's numax; the converged centre becomes the method's numax.

        A numax without a significant comb is rejected and the method refitted without it
        (comb_scan_max_refits); after that the numax is NaN. Also runs the ACV check on the same scans.
        """
        from .comb_scan import recentre
        cfg, name = self.config, method.name
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
        # ACV check (report only): the ACV profile of the last scan of each attempt, in order
        # (the method's estimate, then the refits); the first significant one is kept
        from .comb_scan import acv_significant
        acv = next((a["scans"][-1] for a in attempts if a["scans"]
                    and acv_significant(a["scans"][-1], cfg.comb_scan_acv_threshold)), None)
        acv_numax = acv["acv"]["centre"] if acv else ufloat(np.nan, np.nan)
        if acv and cfg.background_correction:            # like the comb centre
            from .background import correct_numax
            acv_numax = ufloat(correct_numax(spec.frequency, self.running_median(spec),
                                             acv_numax.n), acv_numax.s)
        method.comb_scan = dict(original=original, rejected=rejected, attempts=attempts,
                                accepted=accepted, final=method.numax_override, acv=acv,
                                acv_numax=acv_numax)
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
        """Run the ACF method (numax_2DACF, self.acf)."""
        self.acf = self._run_method(NumaxFromACF, self.acf_config, "numax_2DACF")
        return self.numax_estimates["numax_2DACF"]

    def compute_numax_from_acv(self):
        """Run the ACV method with the ACF_CONFIG settings (numax_ACV, self.acv; not in numax_combined)."""
        self.acv = self._run_method(NumaxFromACV, self.acf_config, "numax_ACV")
        return self.numax_estimates["numax_ACV"]

    def compute_numax_from_CoV(self):
        """Run the CoV method (numax_CoV, self.cov)."""
        self.cov = self._run_method(NumaxFromCoV, self.cov_config, "numax_CoV")
        return self.numax_estimates["numax_CoV"]

    def compute_numax_from_scaling_relations(self):
        """numax from scaling relations with Gaia and/or STAR parameters."""
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
        """Delta-nu from the PSPS around each method's numax: {"dnu_<method>": muHz}."""
        from .artefacts import psps
        cfg, curves = self.config, []
        for label, colour, method, offset in (("ACF", "forestgreen", getattr(self, "acf", None), 0.0),
                                              ("ACV", "dodgerblue", getattr(self, "acv", None), 0.0),
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
        """All estimates as a DataFrame (label, numax, numax_err); saved if CONFIG.save_results."""
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
            if self.entropy_check_results:
                self.entropy_check_table.to_csv(
                    self.out_dir / f"{self.star.target}_entropy_check.txt", index=False)
        return df

    @property
    def cov_scan_table(self) -> pd.DataFrame:
        """CoV scan per method: original and rejected numax, shifted numax, CoV centre, peak z, significant."""
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
    def entropy_check_table(self) -> pd.DataFrame:
        """Entropy check per method, version and setting (bins, peaks per Dnu): numax, best D,
        p-value and significant at the initial and the shifted numax, and the reported numax / D.
        """
        return pd.DataFrame([dict(method=k, **r) for k, sw in self.entropy_check_results.items()
                             for r in sw["rows"]])

    @staticmethod
    def _acv_columns(c) -> dict:
        """ACV-check columns of the comb-scan table for one method."""
        s = c.get("acv") or next((a["scans"][-1] for a in c["attempts"] if a["scans"]), None)
        acv = s["acv"] if s else None
        return {"acv_numax": c["acv_numax"].n,
                "acv_centre": acv["centre"].n if acv else np.nan,
                "acv_numax_err": acv["centre"].s if acv else np.nan,
                "acv_peak_z": acv["peak_z"] if acv else np.nan,
                "acv_significant": c.get("acv") is not None}

    @property
    def comb_scan_table(self) -> pd.DataFrame:
        """Comb scan per method: original and rejected numax, final numax, comb centre, status, peak z
        per profile, D, and the ACV check (acv_numax (background-corrected), acv_centre, acv_peak_z,
        acv_significant).
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
                         "D": last["D"] if a else np.nan,
                         **self._acv_columns(c)})
        return pd.DataFrame(rows)
