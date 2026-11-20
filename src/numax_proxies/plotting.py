"""All figures, one A&A column wide (8.8 cm). Each function saves into
<results_directory>/<target>/figures/ in the formats of CONFIG.figure_format (png, pdf)."""
import functools
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .fitting import gaussian, gaussian_offset_one
from .seismic import cov_upper_limit

try:
    import scienceplots  # noqa: F401
    plt.style.use("science")
except ImportError:  # style is optional
    pass

COLUMN_WIDTH = 8.8 / 2.54          # A&A column [inch]
SIDE_CAPTION_WIDTH = 12.0 / 2.54   # A&A figure with side caption [inch]
DPI = 300                          # png
PDF_RASTER_DPI = 600               # rasterised parts (spectra, 2D ACF) inside pdf files
AA_RC = {                          # applied only while the package draws a figure
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "legend.title_fontsize": 6, "lines.linewidth": 0.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
}


def _aa_style(func):
    """Draw func's figure with AA_RC (and upgreek for an upright mu when LaTeX is used)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rc = dict(AA_RC)
        if plt.rcParams["text.usetex"]:
            preamble = plt.rcParams["text.latex.preamble"]
            if "upgreek" not in preamble:
                rc["text.latex.preamble"] = preamble + r" \usepackage{upgreek}"
        with plt.rc_context(rc):
            return func(*args, **kwargs)
    return wrapper


def _mu() -> str:
    """Upright micro sign: \\upmu under LaTeX, the unicode character otherwise."""
    return r"$\upmu$" if plt.rcParams["text.usetex"] else "μ"


def _freq_label(name: str = "frequency") -> str:
    """Axis label "<name> [μHz]"."""
    return f"{name} [{_mu()}Hz]"


def _psd_label() -> str:
    """Axis label "PSD [ppm²/μHz]"."""
    return f"PSD [ppm$^2$/{_mu()}Hz]"


def _figdir(config, target) -> Path:
    """Figure folder <results_directory>/<target>/figures (created)."""
    path = Path(config.results_directory) / str(target) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, config, target, name: str):
    """Save fig as <name>.<format> for every format in config.figure_format, then close it."""
    folder = _figdir(config, target)
    for fmt in config.figure_format:
        fig.savefig(folder / f"{name}.{fmt}", bbox_inches="tight",
                    dpi=PDF_RASTER_DPI if fmt == "pdf" else DPI)
    plt.close(fig)


def _set_xlim(ax, cfg, fallback_max):
    """Set the x range to [min_freq, max_freq or fallback_max]."""
    ax.set_xlim(cfg.min_freq, cfg.max_freq if cfg.max_freq is not None else fallback_max)


# --------------------------------------------------------------------------------------
@_aa_style
def plot_lightcurve_and_psd(lc, psd, config, target, trend=None, raw_flux=None):
    """Save lc_and_pg: light curve (and Savitzky-Golay trend) and PSD.

    Args:
        lc: LightCurveData.
        psd: PSDData.
        config: ProcessingConfig.
        target: star name.
        trend, raw_flux: Savitzky-Golay filter and flux before detrending (optional).
    """
    fig, axs = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.2))
    if trend is not None:
        axs[0].plot(lc.time, raw_flux, c="k", rasterized=True)
        axs[0].plot(lc.time, trend, c="r", rasterized=True)
        axs[0].text(0.02, 0.97, f"Savitzky-Golay window: {config.savgol_window} d", color="r",
                    ha="left", va="top", transform=axs[0].transAxes, fontsize=6)
    else:
        axs[0].plot(lc.time, lc.flux, c="k", lw=0, ms=1, marker=".", rasterized=True)
    axs[0].set(xlabel="time [days]", ylabel="rel. flux [ppm]", xlim=(lc.time.min(), lc.time.max()))
    axs[1].loglog(psd.frequency, psd.psd, c="k", lw=0.4, rasterized=True)
    axs[1].set(xlabel=_freq_label(), ylabel=_psd_label(),
               xlim=(psd.frequency.min(), psd.frequency.max()))
    fig.tight_layout()
    _save(fig, config, target, "lc_and_pg")


# --------------------------------------------------------------------------------------
def _acf_map_setting(cfg, results):
    """(overlap_scale, width_factor) of the 2D ACF panel (plot_2dacf)."""
    if cfg.plot_2dacf is True:
        scale, factor, _ = sorted(results)[0]
        return scale, factor
    return tuple(cfg.plot_2dacf)


# anticorrelated blue, white noise white, correlated red
ACF_MAP_CMAP = "bwr"


def plot_acf_map(ax, method):
    """Draw the 2D ACF of one grid setting into ax.

    Colour: r(k) / white-noise sd, blue = anticorrelated, white = noise, red = correlated.
    Lag axis: units of Delta-nu(bin center) for collapse delta_nu, muHz for all_lags.

    Args:
        ax: matplotlib axes.
        method: NumaxFromACF after compute().
    """
    from .binning import log_numax_bins
    from .methods.acf import acf_map, band_edges

    cfg = method.cfg
    scale, factor = _acf_map_setting(cfg, method.results)
    freq = np.ascontiguousarray(method.frequency)
    centres, i0, i1 = log_numax_bins(freq, float(max(freq[0], cfg.min_freq)), cfg.max_freq,
                                     factor, scale)
    in_dnu = cfg.collapse == "delta_nu"
    lags, z = acf_map(freq, method.normalized_power, centres, i0, i1,
                      lag_unit="dnu" if in_dnu else "muHz")
    ax.pcolormesh(centres, lags, z, shading="nearest", cmap=ACF_MAP_CMAP, vmin=-4, vmax=4,
                  rasterized=True)
    if in_dnu:
        lo, hi = band_edges(cfg.dnu_multiples, cfg.dnu_window, cfg.dnu_window_mode)
        hi[:-1] = np.minimum(hi[:-1], lo[1:])                    # trimmed like the lag bands
        for edge in np.concatenate([lo, hi]):
            if edge < factor:
                ax.axhline(edge, c="k", ls=":", lw=0.5)
        ax.set(ylabel=r"lag [$\Delta\nu(\nu_\mathrm{c})$]", ylim=(0, factor))
    else:
        ax.set(ylabel=_freq_label("lag"), ylim=(0, lags[-1]))
    ax.text(0.98, 0.96, f"overlap {scale:g}, width {factor:g}", ha="right", va="top",
            transform=ax.transAxes, fontsize=6,
            bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))


def plot_psps_panel(ax, method):
    """Draw the power spectrum of the power spectrum (PSPS) around the method's numax.

    x axis: t x Delta-nu(numax), so a comb gives peaks near 1 and 2 (shaded: the search ranges
    of the comb check); dashed: its threshold.

    Args:
        ax: matplotlib axes.
        method: NumaxFromACF or NumaxFromCoV after compute().
    """
    from .artefacts import psps

    cfg = method.config
    numax = method.numax_estimate.n
    rel = method.normalized_power - (1.0 if method.name == "CoV" else 0.0)
    r = psps(method.frequency, rel, numax, cfg.comb_check_tolerance, cfg.comb_check_fap)
    ax.set(xlabel=r"$t\,\Delta\nu(\nu_\mathrm{max})$", ylabel="PSPS", xlim=(0, 3))
    if r is None:
        ax.text(0.5, 0.5, "no numax", ha="center", va="center", transform=ax.transAxes)
        return
    x = r["t"] * r["dnu"]
    show = x <= 3
    ax.plot(x[show], r["z"][show], c="k", lw=0.6)
    tol = cfg.comb_check_tolerance
    for m in (1.0, 2.0):
        ax.axvspan(m / (1 + tol), m / (1 - tol), color="tab:blue", alpha=0.12, lw=0)
    ax.axhline(r["threshold"], c="r", ls="--", lw=0.6)
    k = np.flatnonzero(r["search"])[np.argmax(r["z"][r["search"]])]
    ax.plot(x[k], r["z"][k], "o", ms=3, mfc="none", mec="r", mew=0.8)
    verdict = (rf"comb, $\Delta\nu \approx$ {r['spacing']:.2f} {_mu()}Hz" if r["comb"]
               else "no comb")
    ax.text(0.98, 0.96, rf"$\nu_\mathrm{{max}}$ = {numax:.1f} {_mu()}Hz" + "\n" + verdict,
            ha="right", va="top", transform=ax.transAxes, fontsize=6,
            bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
    ax.set_ylim(0, max(1.15 * np.max(r["z"][show][1:]), 1.3 * r["threshold"]))


def _draw_spectrum(ax, method):
    """PSD and running median of an ACF/CoV method."""
    ax.loglog(method.frequency, method.power, c="gray", lw=0.4, label="spectrum", rasterized=True)
    ax.loglog(method.frequency, method.med_filter, c="k", label="running median")
    ax.set(xlabel=_freq_label(), ylabel=_psd_label())
    ax.text(0.02, 0.96, method.id, ha="left", va="top", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="lower left")
    _set_xlim(ax, method.cfg, np.max(method.frequency))


def _draw_cacf(ax, method):
    """CACF curves, thresholds and fits of every grid setting."""
    cfg, results = method.cfg, method.results
    for data in results.values():
        bcs = np.asarray(data["bin_centers"])
        ax.plot(bcs, data["cacf_raw"], c="gray", alpha=0.4, lw=0.5, zorder=-3)
        ax.plot(bcs, data["cacf_smoothed"], c="k", alpha=0.4, lw=0.6, zorder=-2)
        ax.plot(bcs, data["fap"], c="green", alpha=0.1, lw=0.6, zorder=0)
        p = data["fit_params"]
        if np.isfinite(p["numax"]):
            x = np.linspace(bcs.min(), bcs.max(), len(bcs) * 10)
            ax.plot(x, gaussian(x, p["amp"], p["sigma"], p["numax"], p["y"]), c="r", alpha=0.4,
                    lw=0.6, zorder=-1)
            ax.axvline(p["numax"], c="b", ls="--", lw=0.5, alpha=0.3, zorder=1)
    handles = [Line2D([0], [0], color="gray", label="raw CACF"),
               Line2D([0], [0], color="k", label="smoothed CACF"),
               Line2D([0], [0], color="r", label="Gaussian fit"),
               Line2D([0], [0], color="green", label="FAP"),
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$")]
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles, loc="upper right")
    ylabel = "CACF" if cfg.collapse == "all_lags" else r"CACF ($\Delta\nu$ lags)"
    ax.set(xlabel=_freq_label(), ylabel=ylabel)


def _right_axis(ax):
    """Ticks and label of the y axis on the right (right column of a 2 x 2 figure)."""
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


@_aa_style
def plot_acf(method):
    """Save ACF: spectrum, CACF curves with thresholds and fits, and optionally the 2D ACF
    (plot_2dacf) and the PSPS around numax (plot_psps). With all four panels: a 2 x 2
    figure 12 cm wide (A&A side caption); otherwise one 8.8 cm column.

    Args:
        method: NumaxFromACF after compute().
    """
    cfg, results = method.cfg, method.results
    show_map, show_psps = cfg.plot_2dacf is not False, bool(cfg.plot_psps)
    if show_map and show_psps:
        fig, a = plt.subplots(2, 2, figsize=(SIDE_CAPTION_WIDTH, 4.4))
        ax_spec, ax_map, ax_cacf, ax_psps = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        _right_axis(ax_map)
        _right_axis(ax_psps)
    else:
        names = ["spec"] + (["map"] if show_map else []) + ["cacf"] + (["psps"] if show_psps else [])
        fig, a = plt.subplots(len(names), 1, figsize=(COLUMN_WIDTH, 2.3 * len(names)))
        panels = dict(zip(names, a))
        ax_spec, ax_cacf = panels["spec"], panels["cacf"]
        ax_map, ax_psps = panels.get("map"), panels.get("psps")

    _draw_spectrum(ax_spec, method)
    _draw_cacf(ax_cacf, method)
    xmax = max(np.max(d["bin_centers"]) for d in results.values())
    for ax in [ax_cacf] + ([ax_map] if show_map else []):
        if cfg.plot_log_scale:
            ax.set_xscale("log")
        _set_xlim(ax, cfg, xmax)
    if show_map:
        plot_acf_map(ax_map, method)
        if cfg.plot_log_scale:
            ax_map.set_xscale("log")
        _set_xlim(ax_map, cfg, xmax)
        ax_map.set_xlabel(_freq_label("bin center"))
    if show_psps:
        plot_psps_panel(ax_psps, method)
    fig.tight_layout()
    _save(fig, method.config, method.id, "ACF")


# --------------------------------------------------------------------------------------
@_aa_style
def plot_cov(method):
    """Save CoVs: CoV curves, thresholds, fits and the CoV upper limit; with plot_psps, the
    PSPS around numax below.

    Args:
        method: NumaxFromCoV after compute().
    """
    cfg, results = method.cfg, method.results
    if cfg.plot_psps:
        fig, (ax, ax_psps) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.8),
                                          gridspec_kw=dict(height_ratios=[1.4, 1]))
    else:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))
    alpha = 0.25
    for data in results.values():
        bcs = np.asarray(data["bin_centers"])
        ax.plot(bcs, data["cov"], c="gray", alpha=alpha, lw=0.5, zorder=-5)
        ax.plot(bcs, data["cov_smoothed"], c="k", alpha=alpha, lw=0.6, zorder=-4)
        ax.plot(bcs, data["fap"], c="green", alpha=alpha, lw=0.6, zorder=-3)
        p = data["fit_params"]
        if all(np.isfinite([p["amp"], p["sigma"], p["numax"]])):
            x = np.linspace(bcs.min(), bcs.max(), len(bcs) * 10)
            ax.plot(x, gaussian_offset_one(x, p["amp"], p["sigma"], p["numax"]), c="r",
                    alpha=alpha, lw=0.6, zorder=-2)
            ax.axvline(p["numax"], c="b", ls="--", lw=0.5, alpha=alpha, zorder=-1)
    ymin, ymax = ax.get_ylim()
    x = np.linspace(1e-3, 1e4, 10000)
    ax.fill_between(x, cov_upper_limit(x), 50, color="orange", alpha=0.2, lw=0, zorder=-6)
    ax.set_ylim(ymin, ymax)
    handles = [Line2D([0], [0], color="gray", label="raw CoV"),
               Line2D([0], [0], color="k", label="smoothed CoV"),
               Line2D([0], [0], color="green", label="FAP"),
               Line2D([0], [0], color="r", label="Gaussian fit"),
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$"),
               Line2D([0], [0], color="orange", label="upper limit")]
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles, loc="upper right")
    ax.text(0.98, 0.03, method.id, ha="right", va="bottom", transform=ax.transAxes, fontsize=7)
    ax.set(xlabel=_freq_label(), ylabel="coefficient of variation")
    if cfg.plot_log_scale:
        ax.set_xscale("log")
    _set_xlim(ax, cfg, max(np.max(d["bin_centers"]) for d in results.values()))
    if cfg.plot_psps:
        plot_psps_panel(ax_psps, method)
    fig.tight_layout()
    _save(fig, method.config, method.id, "CoVs")


# --------------------------------------------------------------------------------------
STYLES = {  # label substring -> (line style, colour, legend label)
    "CoV": ((0, (3, 1, 1, 1)), "mediumorchid", r"$\nu_\mathrm{max}^\mathrm{CoV}$"),
    "2DACF": ((0, (5, 1)), "forestgreen", r"$\nu_\mathrm{max}^\mathrm{ACF}$"),
    "SR_logg_teff": ((0, (1, 5)), "dodgerblue", r"$\nu_\mathrm{max}^{gT_\mathrm{eff}^{-1/2}}$"),
    "SR": ((0, (1, 5)), "dodgerblue", r"$\nu_\mathrm{max}^\mathrm{SR}$"),
}


@_aa_style
def plot_spectrum_with_all_numax_estimates(psd, star, numax_estimates, config):
    """Save full_spectrum_with_all_estimates.

    Args:
        psd: PSDData.
        star: StarInfo.
        numax_estimates: {label: ufloat}.
        config: ProcessingConfig.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    ax.loglog(psd.frequency, psd.psd, c="gray", lw=0.4, rasterized=True)
    for label, numax in numax_estimates.items():
        val, err = getattr(numax, "n", numax), getattr(numax, "s", None)
        if not np.isfinite(val):
            continue
        ls, c, name = next((s for key, s in STYLES.items() if key in label),
                           ("-", "k", label))
        text = f"{val:.1f} ± {err:.1f}" if err is not None else f"{val:.1f}"
        ax.axvline(val, linestyle=ls, c=c, label=f"{name}: {text}")
        if err is not None and np.isfinite(err):
            ax.axvspan(val - err, val + err, alpha=0.2, color=c, lw=0)
    ax.set(xlabel=_freq_label(), ylabel=_psd_label(),
           xlim=(psd.frequency.min(), psd.frequency.max()))
    ax.legend(loc="lower left", title=f"{star.target} " + r"$\nu_\mathrm{max}$ " + f"[{_mu()}Hz]")
    fig.tight_layout()
    _save(fig, config, star.target, "full_spectrum_with_all_estimates")
