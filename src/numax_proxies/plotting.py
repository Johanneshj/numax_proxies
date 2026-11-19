"""All figures. Each function saves into <results_directory>/<target>/figures/."""
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


def _figdir(config, target) -> Path:
    """Figure folder <results_directory>/<target>/figures (created)."""
    path = Path(config.results_directory) / str(target) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_xlim(ax, cfg, fallback_max):
    """Set the x range to [min_freq, max_freq or fallback_max]."""
    ax.set_xlim(cfg.min_freq, cfg.max_freq if cfg.max_freq is not None else fallback_max)


# --------------------------------------------------------------------------------------
def plot_lightcurve_and_psd(lc, psd, config, target, trend=None, raw_flux=None):
    """Save lc_and_pg.png: light curve (and Savitzky-Golay trend) and PSD.

    Args:
        lc: LightCurveData.
        psd: PSDData.
        config: ProcessingConfig.
        target: star name.
        trend, raw_flux: Savitzky-Golay filter and flux before detrending (optional).
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    if trend is not None:
        axs[0].plot(lc.time, raw_flux, c="k", lw=2)
        axs[0].plot(lc.time, trend, c="r", lw=2)
        axs[0].text(0.02, 0.98, f"Savitzky-Golay window: {config.savgol_window} d", color="r",
                    ha="left", va="top", transform=axs[0].transAxes)
    else:
        axs[0].plot(lc.time, lc.flux, c="k", lw=0, ms=3, marker=".")
    axs[0].set(xlabel="time [days]", ylabel="rel. flux [ppm]", xlim=(lc.time.min(), lc.time.max()))
    axs[1].loglog(psd.frequency, psd.psd, c="k")
    axs[1].set(xlabel=r"frequency [$\mu$Hz]", ylabel="PSD",
               xlim=(psd.frequency.min(), psd.frequency.max()))
    fig.savefig(_figdir(config, target) / "lc_and_pg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def _acf_map_setting(cfg, results):
    """(overlap_scale, width_factor) of the 2D ACF panel (plot_2dacf)."""
    if cfg.plot_2dacf is True:
        scale, factor, _ = sorted(results)[0]
        return scale, factor
    return tuple(cfg.plot_2dacf)


# anticorrelated purple, white noise black, correlated green
ACF_MAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "purple_black_green", ["#c77dff", "black", "#57e389"])


def plot_acf_map(ax, method):
    """Draw the 2D ACF of one grid setting into ax.

    Colour: r(k) / white-noise sd, purple = anticorrelated, black = noise, green = correlated.
    Lag axis: units of Delta-nu(bin centre) for collapse delta_nu, muHz for all_lags.

    Args:
        ax: matplotlib axes.
        method: NumaxFromACF after compute().
    """
    from .binning import log_numax_bins
    from .methods.acf import acf_map

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
        ms = sorted(cfg.dnu_multiples)
        for j, m in enumerate(ms):
            hi = m * (1 + cfg.dnu_window)
            if j + 1 < len(ms):                                  # trimmed like the lag bands
                hi = min(hi, ms[j + 1] * (1 - cfg.dnu_window))
            for edge in (m * (1 - cfg.dnu_window), hi):
                if edge < factor:
                    ax.axhline(edge, c="w", ls=":", lw=0.8)
        ax.set(ylabel=r"lag [$\Delta\nu(\nu_c)$]", ylim=(0, factor))
    else:
        ax.set(ylabel=r"lag [$\mu$Hz]", ylim=(0, lags[-1]))
    ax.text(0.98, 0.95, f"overlap {scale:g}, width {factor:g}", ha="right", va="top",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7, lw=0))


def plot_acf(method):
    """Save ACF.png: spectrum, optional 2D ACF panel, CACF curves, thresholds and fits.

    Args:
        method: NumaxFromACF after compute().
    """
    cfg, results = method.cfg, method.results
    show_map = cfg.plot_2dacf is not False
    fig, axs = plt.subplots(3 if show_map else 2, 1, figsize=(5, 10.5 if show_map else 7))
    if show_map:
        plot_acf_map(axs[1], method)
        axs = [axs[0], axs[2], axs[1]]          # [spectrum, CACF, map]

    ax = axs[0]
    ax.loglog(method.frequency, method.power, c="gray", label="spectrum")
    ax.loglog(method.frequency, method.med_filter, c="k", label="running median")
    ax.set(xlabel=r"frequency [$\mu$Hz]", ylabel="PSD")
    ax.text(0.02, 0.98, method.id, ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc="lower left")
    _set_xlim(ax, cfg, np.max(method.frequency))

    ax = axs[1]
    for data in results.values():
        bcs = np.asarray(data["bin_centers"])
        ax.plot(bcs, data["cacf_raw"], c="gray", alpha=0.4, zorder=-3)
        ax.plot(bcs, data["cacf_smoothed"], c="k", alpha=0.4, zorder=-2)
        ax.plot(bcs, data["fap"], c="green", alpha=0.1, zorder=0)
        p = data["fit_params"]
        if np.isfinite(p["numax"]):
            x = np.linspace(bcs.min(), bcs.max(), len(bcs) * 10)
            ax.plot(x, gaussian(x, p["amp"], p["sigma"], p["numax"], p["y"]), c="r", alpha=0.4, zorder=-1)
            ax.axvline(p["numax"], c="b", ls="--", alpha=0.3, zorder=1)
    handles = [Line2D([0], [0], color="gray", label="raw CACF"),
               Line2D([0], [0], color="k", label="smoothed CACF"),
               Line2D([0], [0], color="r", label="Gaussian fit"),
               Line2D([0], [0], color="green", label="FAP"),
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$")]
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles)
    ylabel = "CACF" if cfg.collapse == "all_lags" else r"CACF ($\Delta\nu$ lags)"
    ax.set(xlabel=r"frequency [$\mu$Hz]", ylabel=ylabel)
    xmax = max(np.max(d["bin_centers"]) for d in results.values())
    for a in ([ax, axs[2]] if show_map else [ax]):
        if cfg.plot_log_scale:
            a.set_xscale("log")
        _set_xlim(a, cfg, xmax)
    if show_map:
        axs[2].set_xlabel(r"bin centre [$\mu$Hz]")

    fig.savefig(_figdir(method.config, method.id) / "ACF.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def plot_cov(method):
    """Save CoVs.png: CoV curves, thresholds, fits and the CoV upper limit.

    Args:
        method: NumaxFromCoV after compute().
    """
    cfg, results = method.cfg, method.results
    fig, ax = plt.subplots(figsize=(5, 4))
    alpha = 0.25
    for data in results.values():
        bcs = np.asarray(data["bin_centers"])
        ax.plot(bcs, data["cov"], c="gray", alpha=alpha, zorder=-5)
        ax.plot(bcs, data["cov_smoothed"], c="k", alpha=alpha, zorder=-4)
        ax.plot(bcs, data["fap"], c="green", alpha=alpha, zorder=-3)
        p = data["fit_params"]
        if all(np.isfinite([p["amp"], p["sigma"], p["numax"]])):
            x = np.linspace(bcs.min(), bcs.max(), len(bcs) * 10)
            ax.plot(x, gaussian_offset_one(x, p["amp"], p["sigma"], p["numax"]), c="r", alpha=alpha, zorder=-2)
            ax.axvline(p["numax"], c="b", ls="--", alpha=alpha, zorder=-1)
    ymin, ymax = ax.get_ylim()
    x = np.linspace(1e-3, 1e4, 10000)
    ax.fill_between(x, cov_upper_limit(x), 50, color="orange", alpha=0.2, zorder=-6)
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
    ax.legend(handles=handles)
    ax.text(0.98, 0.02, method.id, ha="right", va="bottom", transform=ax.transAxes)
    ax.set(xlabel=r"frequency [$\mu$Hz]", ylabel="coefficient of variation")
    if cfg.plot_log_scale:
        ax.set_xscale("log")
    _set_xlim(ax, cfg, max(np.max(d["bin_centers"]) for d in results.values()))
    fig.savefig(_figdir(method.config, method.id) / "CoVs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
STYLES = {  # label substring -> (line style, colour, legend label)
    "CoV": ((0, (3, 1, 1, 1)), "mediumorchid", r"$\nu_\mathrm{max}^\mathrm{CoV}$"),
    "2DACF": ((0, (5, 1)), "forestgreen", r"$\nu_\mathrm{max}^\mathrm{ACF}$"),
    "SR_logg_teff": ((0, (1, 5)), "dodgerblue", r"$\nu_\mathrm{max}^{gT_\mathrm{eff}^{-1/2}}$"),
    "SR": ((0, (1, 5)), "dodgerblue", r"$\nu_\mathrm{max}^\mathrm{SR}$"),
}


def plot_spectrum_with_all_numax_estimates(psd, star, numax_estimates, config):
    """Save full_spectrum_with_all_estimates.png.

    Args:
        psd: PSDData.
        star: StarInfo.
        numax_estimates: {label: ufloat}.
        config: ProcessingConfig.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.loglog(psd.frequency, psd.psd, c="gray")
    for label, numax in numax_estimates.items():
        val, err = getattr(numax, "n", numax), getattr(numax, "s", None)
        if not np.isfinite(val):
            continue
        ls, c, name = next((s for key, s in STYLES.items() if key in label),
                           ("-", "k", label))
        text = f"{val:.1f} ± {err:.1f}" if err is not None else f"{val:.1f}"
        ax.axvline(val, linestyle=ls, c=c, label=f"{name}: {text}")
        if err is not None and np.isfinite(err):
            ax.axvspan(val - err, val + err, alpha=0.2, color=c)
    ax.set(xlabel=r"frequency [$\mu$Hz]", ylabel="PSD",
           xlim=(psd.frequency.min(), psd.frequency.max()))
    ax.legend(loc="lower left", title=f"{star.target} " + r"$\nu_\mathrm{max}$ [$\mu$Hz]")
    fig.savefig(_figdir(config, star.target) / "full_spectrum_with_all_estimates.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
