"""White-noise FAP tables per bin size N.

1. Simulate a pool of independent chi^2_2 points (unit-mean exponential), normalised by
   the pool's median (the pipeline divides by a running median; for white noise ~ln 2).
2. Cut the pool into consecutive, non-overlapping bins of N points.
3. Compute the statistics with the pipeline's own functions:
     ACF : cacf_in_windows on P / median - 1
     CoV : cov_in_windows  on P / median
   Repeat 1-3 with new pools until every N has --n-bins bins.
4. The 95th and 99th percentiles over all bins with N points are the thresholds.
5. Violin plots of the statistic for every N, with the two percentiles.

Memory: one pool of --pool-size points at a time (~0.5 GB for 1e7) plus the statistics
of the current N (4 bytes x n_bins); statistics are written to disk N by N.

    python scripts/make_fap_tables.py                          # writes the package table
    python scripts/make_fap_tables.py --n-bins 1e5 --out-dir test --no-package
    python scripts/make_fap_tables.py --plot-only fap_tables_output   # redraw violins

Outputs
  src/numax_proxies/fap_tables.json       the table the pipeline reads
  <out-dir>/fap_thresholds.csv            the same, as CSV, with n_bins
  <out-dir>/statistics_<method>_N<N>.npy  every simulated statistic (float32)
  <out-dir>/fap_violin.png
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numax_proxies.methods.acf import cacf_in_windows                    # noqa: E402
from numax_proxies.methods.cov import _prefix_sums, cov_in_windows         # noqa: E402

PACKAGE_TABLE = ROOT / "src" / "numax_proxies" / "fap_tables.json"
PERCENTILES = [95.0, 99.0]
METHODS = ("ACF", "CoV")


def simulate_n(N: int, n_bins: int, pool_size: int, rng) -> dict:
    """Statistics of n_bins white-noise bins of N points: {method: float32 array}."""
    pool_size = max(pool_size // N, 1) * N                   # whole bins per pool
    out = {m: np.empty(n_bins, dtype=np.float32) for m in METHODS}
    done = 0
    while done < n_bins:
        k = min(pool_size // N, n_bins - done)               # bins from this pool
        P = rng.exponential(size=k * N)                       # step 1
        x = P / np.median(P)
        del P
        i0 = np.arange(k, dtype=np.int64) * N                 # step 2
        i1 = i0 + N
        out["ACF"][done:done + k] = cacf_in_windows(x - 1.0, i0, i1)      # step 3
        out["CoV"][done:done + k] = cov_in_windows(x, i0, i1, *_prefix_sums(x))
        done += k
    return out


def violins(out_dir: Path, table: pd.DataFrame):                    # step 5
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(METHODS), 1, figsize=(12, 9), dpi=150, sharex=True)
    for ax, method in zip(axs, METHODS):
        t = table[table["method"] == method].sort_values("N")
        Ns = t["N"].tolist()
        pos = np.arange(len(Ns))
        data = []
        for N in Ns:
            s = np.load(out_dir / f"statistics_{method}_N{N}.npy", mmap_mode="r")
            step = max(1, s.size // 200_000)                  # subsample for the KDE
            data.append(np.asarray(s[::step], dtype=float))
        parts = ax.violinplot(data, pos, widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("lightsteelblue")
            b.set_alpha(0.8)
        for p, c in zip(PERCENTILES, ("tab:orange", "tab:red")):
            ax.plot(pos, t[f"p{p:g}"], "o-", color=c, ms=4, lw=1, label=f"{p:g}th percentile")
        ax.set_ylabel(f"{method} statistic per bin")
        ax.legend(fontsize=8)
    axs[-1].set_xticks(pos)
    axs[-1].set_xticklabels([str(N) for N in Ns])
    axs[-1].set_xlabel("N (data points per bin)")
    fig.suptitle(f"white noise, {table['n_bins'].min():.0e} non-overlapping bins per N")
    fig.tight_layout()
    path = out_dir / "fap_violin.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-bins", type=float, default=1e6, help="bins per N")
    ap.add_argument("--pool-size", type=float, default=1e7, help="chi^2_2 points per pool (memory)")
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="fap_tables_output")
    ap.add_argument("--no-package", action="store_true", help="do not overwrite the package table")
    ap.add_argument("--plot-only", metavar="OUT_DIR", help="redraw the violins from OUT_DIR")
    args = ap.parse_args()

    if args.plot_only:
        out_dir = Path(args.plot_only)
        print("wrote", violins(out_dir, pd.read_csv(out_dir / "fap_thresholds.csv")))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Ns = [2 ** k for k in range(int(np.log2(args.n_min)), int(np.log2(args.n_max)) + 1)]
    n_bins, pool = int(args.n_bins), int(args.pool_size)
    seeds = np.random.SeedSequence(args.seed).spawn(len(Ns))
    print(f"N = {Ns[0]} ... {Ns[-1]}, {n_bins:.0e} bins per N, "
          f"{n_bins * sum(Ns):.1e} points in total")

    rows, t0 = [], time.perf_counter()
    for N, ss in zip(Ns, seeds):
        t = time.perf_counter()
        stats = simulate_n(N, n_bins, pool, np.random.default_rng(ss))
        for method, s in stats.items():
            np.save(out_dir / f"statistics_{method}_N{N}.npy", s)
            q = np.percentile(s[np.isfinite(s)], PERCENTILES)         # step 4
            rows.append({"method": method, "N": N, "n_bins": int(np.isfinite(s).sum()),
                         **{f"p{p:g}": float(v) for p, v in zip(PERCENTILES, q)}})
        del stats
        print(f"  N = {N:6d}  {time.perf_counter() - t:7.1f} s   "
              + "   ".join(f"{r['method']} 95/99: {r['p95']:.4f} / {r['p99']:.4f}"
                           for r in rows[-2:]), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "fap_thresholds.csv", index=False, float_format="%.6g")
    if not args.no_package:
        doc = {"description": "Per-bin white-noise thresholds of the RAW statistic for bins "
                              "of N independent chi^2_2 points (scripts/make_fap_tables.py).",
               "created": datetime.date.today().isoformat(),
               "seed": args.seed, "n_bins_per_N": n_bins, "N": Ns}
        for method in METHODS:
            t = table[table["method"] == method].sort_values("N")
            doc[method] = {f"{p:g}": [round(v, 6) for v in t[f"p{p:g}"]] for p in PERCENTILES}
        PACKAGE_TABLE.write_text(json.dumps(doc, indent=1))
        print("wrote", PACKAGE_TABLE)
    print("wrote", out_dir / "fap_thresholds.csv", violins(out_dir, table),
          f"({(time.perf_counter() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
