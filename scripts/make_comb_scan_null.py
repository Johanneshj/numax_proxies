"""White-noise correction of the comb-scan null for line-removed windows (comb_scan_null.json).

The comb scan's null mean and sd are analytic for a window with only its mean removed. This
script simulates chi^2_2 noise with a straight line removed and stores, per profile, the bias
and sd ratio relative to the analytic values on a grid of window lengths, D / n and band
half-widths.

    python scripts/make_comb_scan_null.py          (~5 min)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numax_proxies.comb_scan import PROFILES, _collapse, profile_bands   # noqa: E402

N_GRID = [100, 300, 1000, 3000, 10000, 40000]     # window length [points]
X_GRID = np.round(np.arange(0.04, 0.421, 0.02), 4)  # D / n (2.37 D must fit in the window)
HW_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]       # band half-width [D]
N_SIM = 4000
OUT = Path(__file__).resolve().parents[1] / "src" / "numax_proxies" / "comb_scan_null.json"


def analytic(n, lo, hi, w):
    """Analytic white-noise mean and sd of a profile (mean-removed window)."""
    k = np.arange(n)
    cs = np.concatenate([[0.0], np.cumsum((n - k) / n ** 2)])
    nb = len(lo)
    mean = var0 = 0.0
    for b in range(nb):
        cnt = hi[b] - lo[b] + 1
        s = cs[hi[b] + 1] - cs[lo[b]]
        mean = mean - w[b] * s / cnt / nb
        var0 = var0 + w[b] ** 2 * s / cnt ** 2 / nb ** 2
    return mean, np.sqrt(var0 - mean ** 2 * 2 * n / (n - 1))


def main():
    rng = np.random.default_rng(20261123)
    shape = (len(N_GRID), X_GRID.size, len(HW_GRID))
    out = {name: {"bias": np.zeros(shape), "sd_ratio": np.zeros(shape)} for name in PROFILES}
    for i_n, n in enumerate(N_GRID):
        t0 = time.perf_counter()
        d = np.round(X_GRID * n)
        bands = {(name, hw): profile_bands(d, name, hw) for name in PROFILES for hw in HW_GRID}
        t = np.arange(n) - 0.5 * (n - 1)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        s1 = {key: np.zeros(X_GRID.size) for key in bands}
        s2 = {key: np.zeros(X_GRID.size) for key in bands}
        for _ in range(N_SIM):
            x = rng.exponential(size=n)
            x = x - x.mean() - t * (np.dot(t, x) / np.dot(t, t))
            c = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2, nfft)[:n]
            r = c / c[0]
            for key, (lo, hi, w) in bands.items():
                v = _collapse(r, lo, hi, w)
                s1[key] += v
                s2[key] += v * v
        for (name, hw), (lo, hi, w) in bands.items():
            mu = s1[(name, hw)] / N_SIM
            sd = np.sqrt(s2[(name, hw)] / N_SIM - mu ** 2)
            am, asd = analytic(n, lo, hi, w)
            out[name]["bias"][i_n, :, HW_GRID.index(hw)] = (mu - am) / sd
            out[name]["sd_ratio"][i_n, :, HW_GRID.index(hw)] = sd / asd
        print(f"n = {n:6d}: " + ", ".join(
            f"{name} bias {np.median(out[name]['bias'][i_n]):+.3f} ratio {np.median(out[name]['sd_ratio'][i_n]):.3f}"
            for name in PROFILES) + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    OUT.write_text(json.dumps({"description": __doc__.split("\n\n")[0], "n_sim": N_SIM,
                               "n": N_GRID, "x": X_GRID.tolist(), "half_width": HW_GRID,
                               "profiles": {name: dict(bands=[list(b) for b in PROFILES[name]],
                                                       bias=v["bias"].round(4).tolist(),
                                                       sd_ratio=v["sd_ratio"].round(4).tolist())
                                            for name, v in out.items()}}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
