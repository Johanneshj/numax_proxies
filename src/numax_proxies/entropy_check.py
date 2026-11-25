"""Entropy check (after Suárez 2022): is the power around numax regular, with a spacing near
the expected Delta-nu?

A window of P / median is folded on trial spacings (échelle diagram) and the Shannon entropy of
the folded distribution is computed, for the power ("power") and for the tallest peaks ("peaks").
A spacing is significant if its entropy dip, relative to the star's own median, is deeper than
white noise allows (thresholds in entropy_null.json, else simulated). entropy_sweep runs all
settings at the initial and the shifted numax.
"""
import json
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.signal import find_peaks

from .comb_scan import excess_peak
from .seismic import delta_nu

D_RANGE = (0.7, 1.3)      # trial spacings of the detection [Dnu_s] (as the comb scan)
PLOT_RANGE = (0.5, 1.5)   # trial spacings of the H spectrum and the baseline [Dnu_s]
OVERSAMPLE = 3            # trial steps per échelle-bin drift across the window
N_SIMS = 2000             # white-noise simulations (settings outside entropy_null.json)
PROBS = np.logspace(-3, 0, 31)   # false-alarm levels of the null thresholds
VERSIONS = ("power", "peaks")
MIN_PEAKS = 5             # fewer peaks: the peaks version is not tested
GIANT_NUMAX = 300.0       # below: red giant, peak separation delta-nu_02 ~ 0.122 Dnu + 0.05 muHz
SEP_GIANT = 0.125         # (Huber et al. 2010; the intercept is dropped) [Dnu]
SEP_DWARF = 0.05          # above: no simple delta-nu_02 relation, 0.05 Dnu [Dnu]
NULL_PTS = 200            # points per Dnu of the white-noise spectra of the peaks null


def peak_separation(numax):
    """Minimum separation of the peaks [Dnu]: delta-nu_02 for giants, 0.05 otherwise."""
    return SEP_GIANT if numax < GIANT_NUMAX else SEP_DWARF


def trial_spacings(dnu, width, n_bins, rng=PLOT_RANGE):
    """Trial spacings D: steps of dnu / (OVERSAMPLE n_bins width)."""
    step = dnu / (OVERSAMPLE * n_bins * width)
    return np.arange(rng[0] * dnu, rng[1] * dnu + 0.5 * step, step)


def bin_index(nu, trials, n_bins):
    """Flat échelle-bin index (trial * n_bins + bin) of every frequency, shape (n_trials, n_nu)."""
    k = np.floor((nu[None, :] % trials[:, None]) / trials[:, None] * n_bins).astype(int)
    return np.minimum(k, n_bins - 1) + n_bins * np.arange(trials.size)[:, None]


def folded(k, n_bins, sums, counts=None):
    """Folded distribution q (n_trials, n_bins) from bin_index k: mean per bin (counts given) or sum."""
    size = k.size // k.shape[1] * n_bins
    s = np.bincount(k.ravel(), np.broadcast_to(sums, k.shape).ravel(), size).reshape(-1, n_bins)
    if counts is not None:
        c = np.bincount(k.ravel(), np.broadcast_to(counts, k.shape).ravel(), size)
        s = s / np.maximum(c.reshape(-1, n_bins), 1)
    return s / s.sum(axis=1, keepdims=True)


def fold_exact(f, x, trials, n_bins, cs=None):
    """Folded distribution q (n_trials, n_bins): mean of x per échelle bin, every point at its
    own frequency (sums between the bin edges from the running sum cs of x)."""
    cs = np.concatenate([[0.0], np.cumsum(x)]) if cs is None else cs
    width = trials / n_bins                                   # échelle-bin width per trial
    m0 = np.ceil(f[0] / width).astype(int)                    # first edge inside the window
    m1 = np.floor(f[-1] / width).astype(int)
    m = m0[:, None] + np.arange(int((m1 - m0).max()) + 1)[None, :]
    edge = np.where(m <= m1[:, None], m * width[:, None], np.inf)
    idx = np.searchsorted(f, edge.ravel()).reshape(edge.shape)            # first point >= edge
    bounds = np.column_stack([np.zeros(trials.size, int), idx, np.full(trials.size, f.size)])
    s = cs[bounds[:, 1:]] - cs[bounds[:, :-1]]                # sum of x per stretch
    c = np.diff(bounds, axis=1)                               # points per stretch
    k = (m0[:, None] - 1 + np.arange(bounds.shape[1] - 1)[None, :]) % n_bins
    flat = (k + n_bins * np.arange(trials.size)[:, None]).ravel()
    size = trials.size * n_bins
    q = (np.bincount(flat, s.ravel(), size) / np.maximum(np.bincount(flat, c.ravel(), size), 1)
         ).reshape(-1, n_bins)
    return q / q.sum(axis=1, keepdims=True)


def entropy(q):
    """Shannon entropy in bits along the last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(q > 0, q * np.log2(q), 0.0), axis=-1)


def g_statistic(h, n, n_bins):
    """G = 2 ln2 n (log2 n_bins - H)."""
    return 2.0 * np.log(2.0) * n * (np.log2(n_bins) - h)


def tallest_peaks(frequency, p_med, n_peaks, distance, gate=None):
    """Frequencies of the n_peaks tallest local maxima at least distance points apart (with gate:
    only those above a noise cut that about gate noise peaks pass per window)."""
    pk, prop = find_peaks(p_med, height=0, distance=max(1, int(distance)))
    h = prop["peak_heights"]
    if gate is not None:
        ok = h * np.log(2.0) > np.log(p_med.size / gate)
        pk, h = pk[ok], h[ok]
    top = pk[np.argsort(h)[::-1][:n_peaks]]
    return np.sort(frequency[top])


def _null(g, inr):
    """Null summary from simulated G: the contrast threshold for every false-alarm level in PROBS,
    and the mean G."""
    c = np.sort(g[:, inr].max(axis=1) / np.median(g, axis=1))
    u = np.unique(c)
    surv = 1.0 - np.searchsorted(c, u) / c.size             # fraction >= u
    q = [float(u[surv <= p].min()) if (surv <= p).any() else np.inf for p in PROBS]
    return dict(q=np.array(q), mean_g=float(g.mean()))


def threshold(null, fap):
    """Contrast threshold at false-alarm level fap (log-interpolated between PROBS)."""
    q = np.where(np.isfinite(null["q"]), null["q"], np.nan)
    ok = np.isfinite(q)
    return float(np.interp(np.log(fap), np.log(PROBS[ok]), q[ok]))


def p_value(null, contrast):
    """Smallest level p in PROBS whose threshold the contrast reaches (1 if none)."""
    hit = np.flatnonzero(null["q"] <= contrast)
    return float(PROBS[hit].min()) if hit.size else 1.0


@lru_cache(maxsize=1)
def _null_table():
    """Precomputed null summaries (entropy_null.json), or {} if the file is missing."""
    try:
        return json.loads(resources.files("numax_proxies").joinpath("entropy_null.json").read_text())
    except FileNotFoundError:
        return {}


def _from_table(key):
    """Null summary from the table, or None."""
    t = _null_table()
    if (t.get("meta") != dict(d_range=list(D_RANGE), plot_range=list(PLOT_RANGE),
                              oversample=OVERSAMPLE, probs=len(PROBS)) or key not in t):
        return None
    return dict(q=np.array([np.inf if v is None else v for v in t[key]["q"]]),
                mean_g=t[key]["mean_g"])


def _null_trials(n_bins, width):
    """Trial spacings of the simulations (Dnu_s = 1) and the D_RANGE mask."""
    t = trial_spacings(1.0, width, n_bins)
    return t, (t >= D_RANGE[0]) & (t <= D_RANGE[1])


def null_power(n_bins, width, n_sims=N_SIMS, seed=1):
    """White-noise null summary of the power version (table, else simulated)."""
    return _from_table(f"power|{n_bins}|{width:g}") or _sim_power(n_bins, width, n_sims, seed)


@lru_cache(maxsize=32)
def _sim_power(n_bins, width, n_sims=N_SIMS, seed=1):
    """Simulated null summary of the power version (white noise, 32 points per échelle bin)."""
    rng = np.random.default_rng(seed)
    trials, inr = _null_trials(n_bins, width)
    f = 50.0 + np.arange(int(round(width * n_bins * 32))) / (n_bins * 32)
    g = np.empty((n_sims, trials.size))
    for i in range(n_sims):
        g[i] = g_statistic(entropy(fold_exact(f, rng.exponential(size=f.size), trials, n_bins)),
                           f.size, n_bins)
    return _null(g, inr)


def null_peaks(n_bins, width, n_peaks, sep, n_sims=N_SIMS, seed=2):
    """White-noise null summary of the peaks version (table, else simulated)."""
    return (_from_table(f"peaks|{n_bins}|{width:g}|{n_peaks}|{sep:g}")
            or _sim_peaks(n_bins, width, n_peaks, sep, n_sims, seed))


@lru_cache(maxsize=256)
def _sim_peaks(n_bins, width, n_peaks, sep, n_sims=N_SIMS, seed=2):
    """Simulated null summary of the peaks version (tallest peaks of white-noise spectra)."""
    rng = np.random.default_rng(seed)
    trials, inr = _null_trials(n_bins, width)
    f = 50.0 + np.arange(int(round(width * NULL_PTS))) / NULL_PTS
    g = np.empty((n_sims, trials.size))
    for i in range(n_sims):
        pk = tallest_peaks(f, rng.exponential(size=f.size), n_peaks, sep * NULL_PTS)
        g[i] = g_statistic(entropy(folded(bin_index(pk, trials, n_bins), n_bins,
                                          np.ones(pk.size))), pk.size, n_bins)
    return _null(g, inr)


def window(frequency, rel_power, numax, width=8.0, span=2.0, aim_at="excess"):
    """Window of width x Dnu_s around the (shifted) numax, or None if it does not fit."""
    frequency = np.asarray(frequency, dtype=float)
    if not np.isfinite(numax) or numax <= 0:
        return None
    shifted = (excess_peak(frequency, rel_power, float(numax), span) if aim_at == "excess"
               else float(numax))
    if not np.isfinite(shifted):
        return None
    dnu = float(delta_nu(shifted))
    lo, hi = shifted - 0.5 * width * dnu, shifted + 0.5 * width * dnu
    if lo < frequency[0] or hi > frequency[-1]:
        return None
    sel = slice(*np.searchsorted(frequency, [lo, hi]))
    rel = np.asarray(rel_power)[sel]                          # views of the input arrays
    show = slice(*np.searchsorted(frequency, [lo - PLOT_RANGE[1] * dnu, hi + PLOT_RANGE[1] * dnu]))
    return dict(shifted=shifted, dnu=dnu, f=frequency[sel], rel=rel, p_med=rel + 1.0,
                window=(lo, hi),                     # display: whole échelle rows (plots only)
                display=(frequency[show], np.asarray(rel_power)[show]))


def prepare(frequency, rel_power, numax, width=8.0, span=2.0, aim_at="excess"):
    """Window plus everything reused by all settings: peak candidates and the running sum."""
    w = window(frequency, rel_power, numax, width, span, aim_at)
    if w is None:
        return None
    df = w["f"][1] - w["f"][0]
    sep = peak_separation(w["shifted"])
    pk, prop = find_peaks(w["p_med"], height=0, distance=max(1, int(sep * w["dnu"] / df)))
    order = np.argsort(prop["peak_heights"])[::-1]
    w.update(numax=float(numax), width=float(width), df=df, sep=sep, cand=pk[order],
             cand_h=prop["peak_heights"][order],
             cs=np.concatenate([[0.0], np.cumsum(w["p_med"])]))  # running sum (fold_exact)
    return w


def select_peaks(w, n_peaks, gate):
    """The n_peaks tallest candidates (above the noise cut if gate), sorted by frequency."""
    pk, h = w["cand"], w["cand_h"]
    if gate is not None:
        pk = pk[h * np.log(2.0) > np.log(w["p_med"].size / gate)]
    return np.sort(w["f"][pk[:n_peaks]])


def evaluate(w, n_bins, peaks, fap=0.01, power=True):
    """Entropy spectrum, best spacing and significance of a prepared window, per version; None if
    the window is too short."""
    f, dnu, width = w["f"], w["dnu"], w["width"]
    if f.size < 10 * n_bins:
        return None
    trials = trial_spacings(dnu, width, n_bins)
    inr = (trials >= D_RANGE[0] * dnu) & (trials <= D_RANGE[1] * dnu)
    out = dict(numax=w["numax"], scan_centre=w["shifted"], dnu=dnu, width=width, n_bins=n_bins,
               window=w["window"], trials=trials, peak_freqs=peaks,
               spectrum=(f, w["rel"]), display=w["display"])  # views: P / median - 1
    null, versions = {}, {}
    if power:                                    # every point at its own frequency
        null["power"] = null_power(n_bins, width)
        versions["power"] = (fold_exact(f, w["p_med"], trials, n_bins, w["cs"]), f.size)
    if peaks.size >= MIN_PEAKS:
        null["peaks"] = null_peaks(n_bins, width, peaks.size, w["sep"])
        versions["peaks"] = (folded(bin_index(peaks, trials, n_bins), n_bins,
                                    np.ones(peaks.size)), peaks.size)
    else:
        out["peaks"] = dict(H=np.full(trials.size, np.nan), G=np.full(trials.size, np.nan),
                            n=int(peaks.size), D=np.nan, baseline=np.nan, contrast=np.nan,
                            threshold=np.nan, p_value=np.nan, H_baseline=np.nan,
                            H_threshold=np.nan, significant=False)
    for v, (q, n) in versions.items():
        h = entropy(q)
        g = g_statistic(h, n, n_bins)
        j = int(np.flatnonzero(inr)[np.argmax(g[inr])])
        base = float(np.median(g))
        thr = threshold(null[v], fap)
        to_h = 2 * np.log(2) * n                              # G -> log2(n_bins) - H
        out[v] = dict(H=h, G=g, n=int(n), D=float(trials[j]), baseline=base,
                      contrast=float(g[j] / base), threshold=thr,
                      p_value=p_value(null[v], g[j] / base),
                      H_baseline=float(np.log2(n_bins) - base / to_h),
                      H_threshold=float(np.log2(n_bins) - thr * base / to_h),
                      significant=bool(g[j] / base >= thr))
    return out


def entropy_check(frequency, rel_power, numax, width=8.0, n_bins=10, peaks_per_dnu=2.0, gate=None,
                  fap=0.01, span=2.0, aim_at="excess"):
    """One entropy check around numax (shifted to the power excess unless aim_at != "excess"),
    with the tallest peaks_per_dnu x width peaks; None if the window does not fit."""
    w = prepare(frequency, rel_power, numax, width, span, aim_at)
    if w is None:
        return None
    return evaluate(w, n_bins, select_peaks(w, int(round(peaks_per_dnu * width)), gate), fap)


def peak_numbers(width, peaks):
    """Numbers of peaks for a window: every integer from min(peaks) to max(peaks) x width."""
    return list(range(int(round(min(peaks) * width)), int(round(max(peaks) * width)) + 1))


def window_widths(numax, boundary=300.0, power=(8.0, 10.0), peaks=(6.0, 8.0)):
    """Window widths [Dnu] of the two versions: first value below the boundary numax, second above."""
    i = 0 if numax < boundary else 1
    return dict(power=float(power[i]), peaks=float(peaks[i]))


def entropy_sweep(frequency, rel_power, numax, widths=None, bins=tuple(range(5, 16)),
                  peaks=(2.0, 3.0), gates=(None,), fap=0.01, span=2.0):
    """Entropy check for every setting (bins; number of peaks, from the range peaks per Dnu; noise
    cut) at the initial and the shifted numax, each version in its own window width.

    Per setting the reported numax is the initial one if significant, else the shifted one.
    Returns dict(checks, rows, widths).
    """
    widths = window_widths(numax) if widths is None else widths
    checks, rows = {}, []
    for v in VERSIONS:
        width = widths[v]
        settings = ([(n, g) for n in peak_numbers(width, peaks) for g in gates] if v == "peaks"
                    else [(None, None)])
        for aim in ("none", "excess"):
            w = prepare(frequency, rel_power, numax, width, span, aim)
            sel = {(n, g): None if w is None or n is None else select_peaks(w, n, g)
                   for n, g in settings}
            for nb in bins:
                for n, g in settings:
                    checks[(v, aim, nb, n, g)] = None if w is None else evaluate(
                        w, nb, sel[(n, g)] if v == "peaks" else np.array([]), fap,
                        power=v == "power")
        for nb in bins:
            for n, g in settings:
                row = dict(width=width, version=v, bins=nb, peaks=n if v == "peaks" else np.nan,
                           n_peaks=np.nan,
                           noise_cut=("none" if g is None else g) if v == "peaks" else np.nan)
                rep = None
                for aim, tag in (("none", "initial"), ("excess", "shifted")):
                    c = checks[(v, aim, nb, n, g)]
                    e = c[v] if c else {}
                    row.update({f"numax_{tag}": c["scan_centre"] if c else np.nan,
                                f"D_{tag}": e.get("D", np.nan),
                                f"p_{tag}": e.get("p_value", np.nan),
                                f"contrast_{tag}": e.get("contrast", np.nan),
                                f"significant_{tag}": bool(e.get("significant", False))})
                    if rep is None and e.get("significant", False):
                        rep = (tag, c["scan_centre"], e["D"], e["n"])
                row.update(reported=rep[0] if rep else "none",
                           numax=rep[1] if rep else np.nan, D=rep[2] if rep else np.nan)
                if v == "peaks" and rep:
                    row["n_peaks"] = rep[3]
                rows.append(row)
    return dict(checks=checks, rows=rows, widths=widths)
