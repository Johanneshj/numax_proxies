"""All figures (A&A column width), saved to <results_directory>/<target>/figures/ in CONFIG.figure_format."""
import functools
from pathlib import Path

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


def _pct() -> str:
    """Percent sign that also works under LaTeX."""
    return r"\%" if plt.rcParams["text.usetex"] else "%"


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
    """Save lc_and_pg: light curve (with the Savitzky-Golay trend, if any) and PSD."""
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
    """Draw the 2D ACF of one grid setting (r / white-noise sd; the raw ACV for the ACV method)."""
    from .binning import log_numax_bins
    from .methods.acf import acf_map, band_edges

    cfg = method.cfg
    scale, factor = _acf_map_setting(cfg, method.results)
    freq = np.ascontiguousarray(method.frequency)
    centres, i0, i1 = log_numax_bins(freq, float(max(freq[0], cfg.min_freq)), cfg.max_freq,
                                     factor, scale)
    in_dnu = cfg.collapse == "delta_nu"
    acv = cfg.collapse == "autocovariance"
    lags, z = acf_map(freq, method.normalized_power, centres, i0, i1,
                      lag_unit="dnu" if in_dnu else "muHz", normalise=not acv)
    vmax = float(np.nanpercentile(np.abs(z), 99)) if acv else 4.0
    ax.pcolormesh(centres, lags, z, shading="nearest", cmap=ACF_MAP_CMAP, vmin=-vmax, vmax=vmax,
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
    label = f"overlap {scale:g}, width {factor:g}"
    if acv:
        label += f"\nACV, colour $\\pm${vmax:.2g}"
    ax.text(0.98, 0.96, label, ha="right", va="top",
            transform=ax.transAxes, fontsize=6,
            bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))


def plot_comb_check_panel(ax, method):
    """Draw the comb check (PSPS around numax, search ranges, threshold, Delta-nu)."""
    from .artefacts import psps

    cfg = method.config
    numax = method.numax_estimate.n
    rel = method.normalized_power - (1.0 if method.name == "CoV" else 0.0)
    r = psps(method.frequency, rel, numax, cfg.comb_check_tolerance, cfg.comb_check_fap,
             oversampling=cfg.psps_oversampling)
    ax.set(xlabel=_freq_label("$2/t$"), ylabel=r"PS $\otimes$ PS")
    if r is None:
        ax.text(0.5, 0.5, "no numax", ha="center", va="center", transform=ax.transAxes)
        return
    lo, hi = 0.5 * r["dnu"], 2.6 * r["dnu"]
    good = r["t_fine"] > 0
    x_f, z_f = 2.0 / r["t_fine"][good], r["z_fine"][good]
    show_f = (x_f >= lo) & (x_f <= hi)
    ax.plot(x_f[show_f], z_f[show_f], c="k", lw=0.5)
    good = r["t"] > 0
    x, z = 2.0 / r["t"][good], r["z"][good]
    show = (x >= lo) & (x <= hi)
    ax.plot(x[show], z[show], "o", c="k", ms=1.2, mew=0)
    tol = cfg.comb_check_tolerance
    for m in (1.0, 2.0):
        ax.axvspan(m * r["dnu"] * (1 - tol), m * r["dnu"] * (1 + tol), color="tab:blue",
                   alpha=0.12, lw=0)
    ax.axhline(r["threshold"], c="r", ls="--", lw=0.6)
    k = np.flatnonzero(r["search"])[np.argmax(r["z"][r["search"]])]
    ax.plot(2.0 / r["t"][k], r["z"][k], "o", ms=3, mfc="none", mec="r", mew=0.8)
    if r["comb"]:
        ax.axvline(r["spacing"], c="0.3", ls=":", lw=0.6)
    verdict = (rf"comb, $\Delta\nu \approx$ {r['spacing']:.2f} {_mu()}Hz" if r["comb"]
               else "no comb")
    ax.text(0.98, 0.96, rf"$\nu_\mathrm{{max}}$ = {numax:.1f} {_mu()}Hz" + "\n" + verdict,
            ha="right", va="top", transform=ax.transAxes, fontsize=6,
            bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
    ax.set(xlim=(lo, hi), ylim=(0, max(1.15 * np.max(z_f[show_f]), 1.3 * r["threshold"])))


def _right_axis(ax):
    """Ticks and label of the y axis on the right (right column of a 2 x 2 figure)."""
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def _draw_spectrum(ax, method):
    """PSD and running median of a method."""
    ax.loglog(method.frequency, method.power, c="gray", lw=0.4, label="spectrum", rasterized=True)
    ax.loglog(method.frequency, method.med_filter, c="k", label="running median")
    ax.set(xlabel=_freq_label(), ylabel=_psd_label())
    ax.text(0.02, 0.96, method.id, ha="left", va="top", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="lower left")
    _set_xlim(ax, method.cfg, np.max(method.frequency))


def _draw_final(ax, method, handles):
    """Line at the method's final numax."""
    final = method.numax_estimate.n
    if np.isfinite(final):
        ax.axvline(final, c="m", lw=1.2, zorder=2)
        handles.append(Line2D([0], [0], color="m", lw=1.2,
                              label=rf"final $\nu_\mathrm{{max}}$ = {final:.1f}"))


def _draw_cacf(ax, method):
    """Collapsed-statistic curves, thresholds and fits of every grid setting."""
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
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$ per setting")]
    _draw_final(ax, method, handles)
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles, loc="upper right")
    ylabel = {"all_lags": "CACF", "autocovariance": r"mean $|$ACV$|$"}.get(
        cfg.collapse, r"CACF ($\Delta\nu$ lags)")
    ax.set(xlabel=_freq_label(), ylabel=ylabel)


@_aa_style
def plot_acf(method):
    """Save <name>.png for the ACF / ACV method: spectrum, curves and fits, optional 2D ACF and comb check."""
    cfg, results = method.cfg, method.results
    show_map, show_wk = cfg.plot_2dacf is not False, bool(cfg.plot_comb_check)
    if show_map and show_wk:
        fig, a = plt.subplots(2, 2, figsize=(SIDE_CAPTION_WIDTH, 4.4))
        panels = {"spec": a[0, 0], "map": a[0, 1], "cacf": a[1, 0], "wk": a[1, 1]}
        _right_axis(panels["map"])
        _right_axis(panels["wk"])
        names = ["spec", "map", "cacf"]
    else:
        names = ["spec"] + (["map"] if show_map else []) + ["cacf"] + (["wk"] if show_wk else [])
        fig, a = plt.subplots(len(names), 1, figsize=(COLUMN_WIDTH, 2.3 * len(names)))
        panels = dict(zip(names, a))
        names = [n for n in names if n != "wk"]
    _draw_spectrum(panels["spec"], method)
    _draw_cacf(panels["cacf"], method)
    xmax = max(np.max(d["bin_centers"]) for d in results.values())
    for name in names[1:]:
        if cfg.plot_log_scale:
            panels[name].set_xscale("log")
        _set_xlim(panels[name], cfg, xmax)
    if show_map:
        plot_acf_map(panels["map"], method)
        _set_xlim(panels["map"], cfg, xmax)
        panels["map"].set_xlabel(_freq_label("bin center"))
    if show_wk:
        plot_comb_check_panel(panels["wk"], method)
    fig.tight_layout()
    _save(fig, method.config, method.id, method.name)


# --------------------------------------------------------------------------------------
@_aa_style
def plot_cov(method):
    """Save CoVs.png: CoV curves, thresholds, fits and the CoV upper limit, optional comb check."""
    cfg, results = method.cfg, method.results
    if cfg.plot_comb_check:
        fig, (ax, ax_wk) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.8),
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
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$ per setting"),
               Line2D([0], [0], color="orange", label="upper limit")]
    _draw_final(ax, method, handles)
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles, loc="upper right")
    ax.text(0.98, 0.03, method.id, ha="right", va="bottom", transform=ax.transAxes, fontsize=7)
    ax.set(xlabel=_freq_label(), ylabel="coefficient of variation")
    if cfg.plot_log_scale:
        ax.set_xscale("log")
    _set_xlim(ax, cfg, max(np.max(d["bin_centers"]) for d in results.values()))
    if cfg.plot_comb_check:
        plot_comb_check_panel(ax_wk, method)
    fig.tight_layout()
    _save(fig, method.config, method.id, "CoVs")


# --------------------------------------------------------------------------------------
STYLES = {  # label substring -> (line style, colour, legend label); first match wins
    "combined": ("-", "k", r"$\nu_\mathrm{max}$ (mean)"),
    "CoVscan_ACF": ((0, (1, 1)), "darkorange", r"$\nu_\mathrm{max}^\mathrm{CoV\,scan}$ (ACF)"),
    "CoVscan_CoV": ((0, (1, 1)), "saddlebrown", r"$\nu_\mathrm{max}^\mathrm{CoV\,scan}$ (CoV)"),
    "CoVscan_ACV": ((0, (1, 1)), "cadetblue", r"$\nu_\mathrm{max}^\mathrm{CoV\,scan}$ (ACV)"),
    "ACV": ((0, (5, 1, 1, 1)), "dodgerblue", r"$\nu_\mathrm{max}^\mathrm{ACV}$"),
    "CoV": ((0, (3, 1, 1, 1)), "mediumorchid", r"$\nu_\mathrm{max}^\mathrm{CoV}$"),
    "2DACF": ((0, (5, 1)), "forestgreen", r"$\nu_\mathrm{max}^\mathrm{ACF}$"),
    "SR_logg_teff": ((0, (1, 5)), "dimgray", r"$\nu_\mathrm{max}^{gT_\mathrm{eff}^{-1/2}}$"),
    "SR": ((0, (1, 5)), "dimgray", r"$\nu_\mathrm{max}^\mathrm{SR}$"),
}


@_aa_style
def plot_spectrum_with_all_numax_estimates(psd, star, numax_estimates, config):
    """Save full_spectrum_with_all_estimates: the PSD with every numax estimate."""
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


# --------------------------------------------------------------------------------------
@_aa_style
def plot_psps(curves, config, target):
    """Save PSPS.png: the PSPS around each method's numax, with the Delta-nu estimate."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.8))
    d_max = max(c[3]["dnu"] for c in curves)
    lo, hi = 0.4 * d_max, 2.8 * d_max
    for label, colour, numax, r in curves:
        good = r["t_fine"] > 0
        x = 2.0 / r["t_fine"][good]
        show = (x >= lo) & (x <= hi)
        ax.plot(x[show], r["z_fine"][good][show], c=colour, lw=0.7,
                label=rf"{label}: $\nu_\mathrm{{max}}$ = {numax:.1f}, "
                      rf"$\Delta\nu$ = {r['spacing']:.2f} {_mu()}Hz")
        ax.axvline(r["spacing"], c=colour, ls=":", lw=0.7)
    ax.legend(loc="upper right")
    ax.set(xlabel=_freq_label("$2/t$"), ylabel=r"PS $\otimes$ PS", xlim=(lo, hi))
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.96, str(target), ha="left", va="top", transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    _save(fig, config, target, "PSPS")


# --------------------------------------------------------------------------------------
@_aa_style
def plot_comb_scan(r, rel_power, frequency, name, config, target):
    """Save CombScan_<name>.png: spectrum segment, local 2D ACF with the lag bands, and the
    profiles with the comb fit and the ACV check (right axis).
    """
    numax, centres = r["numax"], r["centres"]
    fig, axs = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 5.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.8, 1.1, 1]))
    m = (frequency >= centres[0]) & (frequency <= centres[-1])
    w = max(1, int(round(0.05 * r["dnu"] / (frequency[1] - frequency[0]))))
    ax = axs[0]
    ax.plot(frequency[m], rel_power[m] + 1, c="0.75", lw=0.3, rasterized=True, label="spectrum")
    ax.plot(frequency[m], np.convolve(rel_power[m] + 1, np.ones(w) / w, "same"), c="k", lw=0.6,
            rasterized=True, label="smoothed")
    ax.set(ylabel="P / median", ylim=(0, np.percentile(rel_power[m] + 1, 99.9) * 1.1))

    ax = axs[1]
    ax.imshow(np.clip(r["map"], -4, 4), aspect="auto", origin="lower", cmap="RdBu_r", vmin=-4,
              vmax=4, extent=[centres[0], centres[-1], 0, r["width"]], rasterized=True)
    d = r["D"] / r["dnu"]
    from .comb_scan import HALF, PROFILES
    peaks = {m: HALF for m, _ in PROFILES["combined"]}          # {multiple: half-width}
    dips = {m: HALF for m, _ in PROFILES["anti"]}
    # bands in use: red = correlation, blue = anticorrelation; numbered by position
    # (1 = D/4, 2 = D/2, ..., 9 = 9D/4) so a number means the same lag in every star
    for bands, colour in ((peaks, "red"), (dips, "blue")):
        for mult, hw in bands.items():
            ax.axhspan((mult - hw) * d, (mult + hw) * d, color=colour, alpha=0.18, lw=0)
            for y in ((mult - hw) * d, (mult + hw) * d):          # band edges
                ax.axhline(y, c=colour, lw=0.5)
            ax.text(1.01, mult * d / r["width"], str(int(round(4 * mult))), color=colour,
                    transform=ax.transAxes, ha="left", va="center", fontsize=6)
    ax.set(ylabel=r"lag [$\Delta\nu$]")

    ax = axs[2]
    ax.axhline(0, c="0.5", lw=0.5)
    prof = r["profiles"]
    for key, colour, lw in (("total", "tab:purple", 0.6), ("anti", "tab:cyan", 0.6),
                            ("combined", "0.6", 0.6), ("half", "tab:orange", 0.7),
                            ("integer", "k", 0.9)):
        ax.plot(centres, prof[key]["z"], c=colour, lw=lw, label=key)
    if r["popt"] is not None:
        xs = np.linspace(centres[0], centres[-1], 400)
        ax.plot(xs, gaussian(xs, *r["popt"]), c="tab:red", lw=0.9, label="fit (integer)")
        ax.axvline(r["numax_comb"].n, c="tab:red", lw=0.7, ls=":")
    acv = r.get("acv")
    if acv is not None:                         # ACV check: own axis (z can be much larger)
        ax2 = ax.twinx()
        ax2.plot(centres, acv["z"], c="dodgerblue", lw=0.8)
        if acv["popt"] is not None:
            xs = np.linspace(centres[0], centres[-1], 400)
            ax2.plot(xs, gaussian(xs, *acv["popt"]), c="dodgerblue", lw=0.8, ls="--")
            ax2.axvline(acv["centre"].n, c="dodgerblue", lw=0.7, ls=":")
        ax2.set_ylabel(r"$z_\mathrm{ACV}$", color="dodgerblue")
        ax2.tick_params(axis="y", colors="dodgerblue")
        ax.plot([], [], c="dodgerblue", lw=0.8, label=f"ACV (peak $z$ {acv['peak_z']:.1f})"
                if acv["popt"] is not None else "ACV (no fit)")
    ax.legend(loc="upper right", fontsize=5, frameon=True, framealpha=0.8, edgecolor="none")
    ax.set(xlabel=_freq_label("window centre"), ylabel="$z$")
    for i, ax in enumerate(axs):
        top = i == 0
        ax.axvline(numax, c="tab:blue", lw=0.7, ls="--", label=r"guess $\nu_\mathrm{max}$" if top else None)
        if abs(r.get("scan_centre", numax) - numax) > 1e-6:     # scan re-centred on the excess
            ax.axvline(r["scan_centre"], c="0.4", lw=0.7, ls="-.", label=r"shifted $\nu_\mathrm{max}$" if top else None)
        if r["popt"] is not None and i < 2:
            ax.axvline(r["numax_comb"].n, c="tab:red", lw=0.7, ls=":", label="comb centre" if top else None)
        if r.get("acv") is not None and r["acv"]["popt"] is not None and i < 2:
            ax.axvline(r["acv"]["centre"].n, c="dodgerblue", lw=0.7, ls=":",
                       label="ACV centre" if top else None)
    axs[0].legend(loc="upper right", fontsize=5, frameon=True, framealpha=0.8, edgecolor="none")
    fig.tight_layout(h_pad=0.4)
    _save(fig, config, target, f"CombScan_{name}")



# --------------------------------------------------------------------------------------
@_aa_style
def plot_cov_scan(r, rel_power, frequency, name, config, target):
    """Save CovScan_<name>.png: spectrum segment and the CoV per window with fit and threshold."""
    centres, n = r["centres"], r["n"]
    fig, axs = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 4.0), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1]))
    m = (frequency >= centres[0]) & (frequency <= centres[-1])
    w = max(1, int(round(0.05 * r["dnu"] / (frequency[1] - frequency[0]))))
    ax = axs[0]
    ax.plot(frequency[m], rel_power[m] + 1, c="0.75", lw=0.3, rasterized=True, label="spectrum")
    ax.plot(frequency[m], np.convolve(rel_power[m] + 1, np.ones(w) / w, "same"), c="k", lw=0.6,
            rasterized=True, label="smoothed")
    ax.set(ylabel="P / median", ylim=(0, np.percentile(rel_power[m] + 1, 99.9) * 1.1))

    ax = axs[1]
    mean0, sd0 = 1.0 - 1.0 / n, 1.0 / np.sqrt(n)               # white noise: CoV mean and sd
    ax.axhline(mean0, c="0.5", lw=0.5, label="noise")
    ax.axhline(mean0 + config.cov_scan_threshold * sd0, c="green", lw=0.6, ls="--",
               label="threshold")
    ax.plot(centres, r["cov"], c="k", lw=0.8, label="CoV")
    if r["popt"] is not None:
        xs = np.linspace(centres[0], centres[-1], 400)
        ax.plot(xs, mean0 + gaussian(xs, *r["popt"]) * sd0, c="tab:red", lw=0.9, label="fit")
    ax.set(xlabel=_freq_label("window centre"), ylabel="CoV (window)")
    for i, ax in enumerate(axs):
        top = i == 0
        ax.axvline(r["numax"], c="tab:blue", lw=0.7, ls="--", label=r"guess $\nu_\mathrm{max}$" if top else None)
        ax.axvline(r["scan_centre"], c="0.4", lw=0.7, ls="-.", label=r"shifted $\nu_\mathrm{max}$" if top else None)
        if r["popt"] is not None:
            ax.axvline(r["numax_cov"].n, c="tab:red", lw=0.7, ls=":", label="CoV centre" if top else None)
        ax.legend(loc="upper right", fontsize=5, frameon=True, framealpha=0.8, edgecolor="none")
    fig.tight_layout(h_pad=0.4)
    _save(fig, config, target, f"CovScan_{name}")


# --------------------------------------------------------------------------------------
def _circular_density(phase, weights, counts, n_fine, smooth):
    """Density over [0, 1) of folded phases: fine histogram (mean per bin if counts) smoothed circularly."""
    from scipy.ndimage import gaussian_filter1d
    k = np.minimum((phase * n_fine).astype(int), n_fine - 1)
    h = np.bincount(k, weights, n_fine)
    if counts is not None:
        h = h / np.maximum(np.bincount(k, counts, n_fine), 1)
    h = gaussian_filter1d(h, smooth, mode="wrap")
    return h / h.mean()


ENTROPY_COLOURS = {"power": "#E7298A", "peaks": "#8C510A"}    # pink, brown


@_aa_style
def plot_entropy_check(checks, name, config, target):
    """Save Entropy_<name>.png (one column): the échelle diagram of the power window at the power
    version's best spacing (P / median smoothed over Dnu / 100, the peaks version's peaks circled),
    p_k of both versions (filled bars, with a smoothed curve), and their normalized entropy spectra
    (median 0, FAP threshold -1).
    checks = {version: entropy check} (each version in its own window).
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    from scipy.ndimage import uniform_filter1d
    from matplotlib.colors import to_rgba
    from .entropy_check import bin_index, fold_exact, folded
    leg = dict(fontsize=5, frameon=False, framealpha=0, ncol=1)
    fap = f"{100 * config.entropy_check_fap:g} {_pct()} FAP"
    versions = [v for v in ("power", "peaks") if checks.get(v) and np.isfinite(checks[v][v]["D"])]
    if not versions:
        return
    r = checks[versions[0]]                                  # the échelle window (power first)
    D, dnu, nb = r[versions[0]]["D"], r["dnu"], r["n_bins"]
    fig = plt.figure(figsize=(COLUMN_WIDTH, 13 / 2.54))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.3, 1.0], hspace=0.22)
    top = outer[0].subgridspec(2, 1, height_ratios=[1.6, 0.7], hspace=0)
    ax_e = fig.add_subplot(top[0])
    ax_p = fig.add_subplot(top[1], sharex=ax_e)
    ax_h = fig.add_subplot(outer[1])
    # échelle: whole rows (completed from just outside the window), smoothed for display only
    fd, rd = r["display"]
    df = fd[1] - fd[0]
    smooth = max(1, int(round(dnu / 100 / df)))
    rd = uniform_filter1d(rd, smooth, mode="nearest")
    lo, hi = np.floor(r["window"][0] / D) * D, np.ceil(r["window"][1] / D) * D
    keep = (fd >= lo) & (fd < hi)
    f, rel = fd[keep], rd[keep]
    n_x = max(1, int(round(D / df)))
    row = np.floor(f / D).astype(int)
    row0 = row.min()
    flat = (row - row0) * n_x + np.minimum((f % D / D * n_x).astype(int), n_x - 1)
    size = (row.max() - row0 + 1) * n_x
    img = (np.bincount(flat, rel + 1.0, size) / np.maximum(np.bincount(flat, None, size), 1)
           ).reshape(-1, n_x)
    ax_e.imshow(img, origin="lower", aspect="auto", cmap="Blues", vmin=0,
                vmax=np.percentile(img[img > 0], 99.5), rasterized=True, interpolation="none",
                extent=[0, D, row0 * D, (row.max() + 1) * D])
    pk = checks["peaks"]["peak_freqs"] if "peaks" in versions else np.array([])
    ax_e.plot(pk % D, np.floor(pk / D) * D + 0.5 * D, "o", ms=2.5, mfc="none", mec="orangered",
              mew=0.6, ls="none")
    ax_e.set(ylabel=_freq_label(r"frequency ($\nu$)"), ylim=(row0 * D, (row.max() + 1) * D))
    ax_e.tick_params(labelbottom=False)
    # p_k of both versions on the same spacing, with a smoothed version
    fw, rw = checks["power"]["spectrum"] if "power" in versions else r["spectrum"]
    edges = np.linspace(0, D, nb + 1)
    xs = (np.arange(8 * nb) + 0.5) * D / (8 * nb)
    top_p = 0.0
    for v in versions:
        if v == "power":
            q = fold_exact(fw, rw + 1.0, np.array([D]), nb)[0]
            x, w, c = fw, uniform_filter1d(rw, smooth, mode="nearest") + 1.0, np.ones(fw.size)
        else:
            q = folded(bin_index(pk, np.array([D]), nb), nb, np.ones(pk.size))[0]
            x, w, c = pk, np.ones(pk.size), None
        ax_p.bar(edges[:-1], q, width=D / nb, align="edge", label=v, lw=0.8,   # bins: filled bars
                 color=to_rgba(ENTROPY_COLOURS[v], 0.25), edgecolor=ENTROPY_COLOURS[v])
        ax_p.plot(xs, _circular_density(x % D / D, w, c, 8 * nb, 4.0) / nb,
                  c=ENTROPY_COLOURS[v], lw=0.8)
        top_p = max(top_p, q.max())
    ax_p.set(xlim=(0, D), ylim=(0, 1.45 * top_p), ylabel="$p_k$",
             xlabel=_freq_label(r"$\nu$ mod $\Delta\nu_\mathrm{best}$"))
    ax_p.yaxis.set_major_locator(MaxNLocator(4, prune="upper"))  # no label at the joint
    ax_p.legend(loc="upper right", **leg)
    # normalized entropy spectra: (H - median H) / (median H - FAP level)
    t = r["trials"]
    ax_h.axvline(dnu, c="dimgray", lw=0.6, ls=":", label=r"$\Delta\nu_\mathrm{init}$")
    ax_h.axhline(0, c="k", lw=0.6, zorder=1, label="median $H$")
    ax_h.axhline(-1, c="orangered", lw=0.8, ls="--", zorder=1, label=fap)
    ys = []
    for v in versions:
        e, tv = checks[v][v], checks[v]["trials"]
        y = (e["H"] - e["H_baseline"]) / (e["H_baseline"] - e["H_threshold"])
        ys.append(y)
        ax_h.plot(tv, y, c=ENTROPY_COLOURS[v], lw=0.7, zorder=2, label=v)
        ax_h.plot(e["D"], y[np.argmin(np.abs(tv - e["D"]))], "o", ms=3, c=ENTROPY_COLOURS[v],
                  zorder=3)
    y_lo = min(min(np.nanmin(y) for y in ys), -1.0)
    y_hi = max(np.nanmax(y) for y in ys)
    ax_h.set(xlim=(t[0], t[-1]), ylim=(y_lo - 0.9 * (y_hi - y_lo), y_hi + 0.1 * (y_hi - y_lo)),
             xlabel=_freq_label("trial spacing $D$"), ylabel="normalized $H$")
    ax_h.legend(loc="lower right", **leg)
    _save(fig, config, target, f"Entropy_{name}")
