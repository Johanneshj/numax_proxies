"""White-noise null of the comb-scan statistic, for src/numax_proxies/comb_scan_null.json.

The comb scan collapses each window (relative power minus a straight line) to
    c(D) = 1/2 [ mean r(k) over D/2 +- 0.1 D  +  mean r(k) over D +- 0.1 D ].
For a window with only its MEAN removed, the white-noise mean and sd of c(D) are analytic
(E r(k) = -(n-k)/n^2, Var r(k) = (n-k)/n^2, lags conditioned on the sum rule; as for the
ACF delta_nu collapse). Removing a straight line shifts the mean slightly down and narrows
the spread. This script simulates exponential (chi^2_2) white noise and stores the
correction relative to the analytic values,
    bias  b = (simulated mean - analytic mean) / simulated sd
    ratio s = simulated sd / analytic sd
on a grid of window lengths n and x = D / n (D in frequency points). The comb scan uses
    mean = analytic mean + b x s x analytic sd,   sd = s x analytic sd.

    python scripts/make_comb_scan_null.py          (~2 min)
"""
import json
import time
from pathlib import Path

import numpy as np

N_GRID = [100, 300, 1000, 3000, 10000, 40000]     # window length [points]
X_GRID = np.round(np.arange(0.04, 0.801, 0.02), 4)  # D / n
N_SIM = 4000
HALF = 0.1
OUT = Path(__file__).resolve().parents[1] / "src" / "numax_proxies" / "comb_scan_null.json"


def band_edges(n, x):
    d = np.round(x * n)
    lo = np.array([np.ceil((m - HALF) * d) for m in (0.5, 1.0)]).astype(int)
    hi = np.array([np.floor((m + HALF) * d) for m in (0.5, 1.0)]).astype(int)
    return np.maximum(lo, 1), np.minimum(hi, n - 1)


def collapse(r, lo, hi):
    cs = np.concatenate([[0.0], np.cumsum(r)])
    return 0.5 * sum((cs[hi[b] + 1] - cs[lo[b]]) / (hi[b] - lo[b] + 1) for b in range(2))


def analytic(n, lo, hi):
    k = np.arange(n)
    cs = np.concatenate([[0.0], np.cumsum((n - k) / n ** 2)])
    mean = var0 = 0.0
    for b in range(2):
        cnt = hi[b] - lo[b] + 1
        s = cs[hi[b] + 1] - cs[lo[b]]
        mean = mean - 0.5 * s / cnt
        var0 = var0 + 0.25 * s / cnt ** 2
    return mean, np.sqrt(var0 - mean ** 2 * 2 * n / (n - 1))


def main():
    rng = np.random.default_rng(20261120)
    bias, ratio = [], []
    for n in N_GRID:
        t0 = time.perf_counter()
        lo, hi = band_edges(n, X_GRID)
        i = np.arange(n)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        s1 = np.zeros(X_GRID.size)
        s2 = np.zeros(X_GRID.size)
        for _ in range(N_SIM):
            x = rng.exponential(size=n)
            x = x - np.polyval(np.polyfit(i, x, 1), i)
            c = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:n]
            v = collapse(c / c[0], lo, hi)
            s1 += v
            s2 += v * v
        mu = s1 / N_SIM
        sd = np.sqrt(s2 / N_SIM - mu ** 2)
        am, asd = analytic(n, lo, hi)
        bias.append(((mu - am) / sd).round(4).tolist())
        ratio.append((sd / asd).round(4).tolist())
        print(f"n = {n:6d}: bias {np.median(bias[-1]):+.3f}, sd ratio {np.median(ratio[-1]):.3f} "
              f"(median over x), {time.perf_counter() - t0:.0f} s", flush=True)
    OUT.write_text(json.dumps({"description": __doc__.split("\n\n")[0], "n_sim": N_SIM,
                               "n": N_GRID, "x": X_GRID.tolist(), "bias": bias,
                               "sd_ratio": ratio}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
