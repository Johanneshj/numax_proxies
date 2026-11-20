"""All figures, one A&A column wide (8.8 cm). Each function saves into
<results_directory>/<target>/figures/ in the formats of CONFIG.figure_format (png, pdf)."""
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


def plot_comb_check_panel(ax, method):
    """Draw the comb check (PSPS) around the method's numax.

    PSPS against 2/t, so the strongest peak (period Delta-nu / 2) sits at Delta-nu:
    oversampled (line, psps_oversampling) and the independent points the test uses (dots);
    shaded: the search ranges around Delta-nu and 2 Delta-nu of numax; dashed: the
    threshold; circle: the tested maximum; dotted: the Delta-nu estimate.

    Args:
        ax: matplotlib axes.
        method: NumaxFromACF or NumaxFromCoV after compute().
    """
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
    """PSD and running median of an ACF/CoV method."""
    ax.loglog(method.frequency, method.power, c="gray", lw=0.4, label="spectrum", rasterized=True)
    ax.loglog(method.frequency, method.med_filter, c="k", label="running median")
    ax.set(xlabel=_freq_label(), ylabel=_psd_label())
    ax.text(0.02, 0.96, method.id, ha="left", va="top", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="lower left")
    _set_xlim(ax, method.cfg, np.max(method.frequency))


def _draw_final(ax, method, handles):
    """Solid line at the method's final numax (after comb scan / background correction)."""
    final = method.numax_estimate.n
    if np.isfinite(final):
        ax.axvline(final, c="m", lw=1.2, zorder=2)
        handles.append(Line2D([0], [0], color="m", lw=1.2,
                              label=rf"final $\nu_\mathrm{{max}}$ = {final:.1f}"))


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
               Line2D([0], [0], color="b", ls="--", label=r"$\nu_\mathrm{max}$ per setting")]
    _draw_final(ax, method, handles)
    if method.initial_numax:
        ax.axvline(method.initial_numax, c="green", ls="-.")
        handles.append(Line2D([0], [0], color="green", ls="-.", label="initial guess"))
    ax.legend(handles=handles, loc="upper right")
    ylabel = "CACF" if cfg.collapse == "all_lags" else r"CACF ($\Delta\nu$ lags)"
    ax.set(xlabel=_freq_label(), ylabel=ylabel)


@_aa_style
def plot_acf(method):
    """Save ACF: spectrum, CACF curves with thresholds and fits, and optionally the 2D ACF
    (plot_2dacf) and the comb check (plot_comb_check). With all four panels: a 2 x 2 figure
    12 cm wide (A&A side caption), y axes of the right column on the right; otherwise one
    8.8 cm column.

    Args:
        method: NumaxFromACF after compute().
    """
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
    _save(fig, method.config, method.id, "ACF")


# --------------------------------------------------------------------------------------
@_aa_style
def plot_cov(method):
    """Save CoVs: CoV curves, thresholds, fits and the CoV upper limit; with plot_comb_check,
    the comb-check diagnostic below.

    Args:
        method: NumaxFromCoV after compute().
    """
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


# --------------------------------------------------------------------------------------
@_aa_style
def plot_psps(curves, config, target):
    """Save PSPS: power spectrum of the power spectrum around each method's numax against
    2/t, so the strongest peak (period Delta-nu / 2, Viani et al. 2019) sits at Delta-nu
    (dotted: the Delta-nu estimate).

    Args:
        curves: [(label, colour, numax [muHz], psps dict from artefacts.psps)].
        config: ProcessingConfig.
        target: star name.
    """
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
def plot_comb_scan(r, rel_power, frequency, name, config, target, note=""):
    """Save CombScan_<name>: one comb scan (see comb_scan.py), dashed blue = the numax it
    was centred on, dotted red = fitted centre.

    Panels: relative power around numax; local 2D ACF (r / sd(r), lag in Delta-nu of the
    tested numax) with the two lag bands; collapsed profile z with the Gaussian fit.

    Args:
        r: dict from comb_scan.comb_scan.
        rel_power: relative power used for the scan.
        frequency: frequency grid [muHz].
        name: "ACF" or "CoV".
        config: ProcessingConfig.
        target: star name.
        note: extra text for the label (e.g. the re-centring status).
    """
    numax, centres = r["numax"], r["centres"]
    pct = r"\%" if plt.rcParams["text.usetex"] else "%"
    fig, axs = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 5.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.8, 1.1, 1]))
    m = (frequency >= centres[0]) & (frequency <= centres[-1])
    w = max(1, int(round(0.05 * r["dnu"] / (frequency[1] - frequency[0]))))
    ax = axs[0]
    ax.plot(frequency[m], rel_power[m] + 1, c="0.75", lw=0.3, rasterized=True)
    ax.plot(frequency[m], np.convolve(rel_power[m] + 1, np.ones(w) / w, "same"), c="k", lw=0.6,
            rasterized=True)
    ax.set(ylabel="P / median", ylim=(0, np.percentile(rel_power[m] + 1, 99.9) * 1.1))
    ax.text(0.02, 0.95, f"{target}, {name}" + (f"\n{note}" if note else ""), transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(facecolor="white", alpha=0.8, lw=0))

    ax = axs[1]
    ax.imshow(np.clip(r["map"], -4, 4), aspect="auto", origin="lower", cmap="RdBu_r", vmin=-4,
              vmax=4, extent=[centres[0], centres[-1], 0, r["width"]], rasterized=True)
    d = r["D"] / r["dnu"]
    for mult in (0.5, 1.0):
        for y in ((mult - 0.1) * d, (mult + 0.1) * d):
            ax.axhline(y, c="k", lw=0.5, ls="--")
    ax.set(ylabel=r"lag [$\Delta\nu$]")

    ax = axs[2]
    ax.axhline(0, c="0.5", lw=0.5)
    ax.plot(centres, r["z"], c="k", lw=0.8)
    if r["popt"] is not None:
        xs = np.linspace(centres[0], centres[-1], 400)
        ax.plot(xs, gaussian(xs, *r["popt"]), c="tab:red", lw=0.9)
        ax.axvline(r["numax_comb"].n, c="tab:red", lw=0.7, ls=":")
        ax.set_title(rf"peak $z$ = {r['peak_z']:.1f}, centre = {r['numax_comb'].n:.1f} "
                     rf"{_mu()}Hz ({100 * r['offset']:+.1f} {pct})", fontsize=7)
    ax.set(xlabel=_freq_label("window centre"), ylabel="$z$")
    for ax in axs:
        ax.axvline(numax, c="tab:blue", lw=0.7, ls="--")
    fig.tight_layout(h_pad=0.4)
    _save(fig, config, target, f"CombScan_{name}")
