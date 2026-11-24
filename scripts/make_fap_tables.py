"""White-noise thresholds per bin size N for src/numax_proxies/fap_tables.json.

Simulates chi^2_2 noise divided by its median, cuts it into bins of N points and computes the
pipeline's own statistics (ACF, ACV, CoV). Stores mu0 (the white-noise mean of the ACF and ACV
statistics) and the 95th and 99th percentiles, and draws violin plots.

    python scripts/make_fap_tables.py                                    # all methods
    python scripts/make_fap_tables.py --methods ACV --max-points 1e10    # one method, merged in
    python scripts/make_fap_tables.py --from-statistics fap_tables_output  # no new simulation
    python scripts/make_fap_tables.py --plot-only fap_tables_output        # violins only
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
from numax_proxies.methods.acf import acv_in_windows, cacf_in_windows    # noqa: E402
from numax_proxies.methods.cov import _prefix_sums, cov_in_windows         # noqa: E402

PACKAGE_TABLE = ROOT / "src" / "numax_proxies" / "fap_tables.json"
PERCENTILES = [95.0, 99.0]
METHODS = ("ACF", "CoV", "ACV")
CENTRED = ("ACF", "ACV")          # thresholds of statistic - mu0


def simulate_n(N: int, n_bins: int, pool_size: int, rng, methods=METHODS) -> dict:
    """Raw statistics of n_bins white-noise bins of N points: {method: array}."""
    pool_size = max(pool_size // N, 1) * N                   # whole bins per pool
    out = {m: np.empty(n_bins, dtype=np.float32) for m in methods}
    done = 0
    while done < n_bins:
        k = min(pool_size // N, n_bins - done)               # bins from this pool
        P = rng.exponential(size=k * N)                       # step 1
        x = P / np.median(P)
        del P
        i0 = np.arange(k, dtype=np.int64) * N                 # step 2
        i1 = i0 + N
        if "ACF" in out:                                      # step 3
            out["ACF"][done:done + k] = cacf_in_windows(x - 1.0, i0, i1)
        if "ACV" in out:
            out["ACV"][done:done + k] = acv_in_windows(x - 1.0, i0, i1)
        if "CoV" in out:
            out["CoV"][done:done + k] = cov_in_windows(x, i0, i1, *_prefix_sums(x))
        done += k
    return out


def used_statistic(method: str, s: np.ndarray) -> tuple[np.ndarray, float]:
    """The statistic as the pipeline uses it (minus mu0 for the ACF and ACV), and mu0."""
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if method in CENTRED:
        mu0 = float(np.mean(s))
        return s - mu0, mu0
    return s, np.nan


def summarise(method: str, N: int, s: np.ndarray) -> dict:
    """mu0 and the 95th / 99th percentiles of one method and N."""
    used, mu0 = used_statistic(method, s)
    q = np.percentile(used, PERCENTILES)
    return {"method": method, "N": N, "n_bins": int(used.size), "mu0": mu0,
            **{f"p{p:g}": float(v) for p, v in zip(PERCENTILES, q)}}


def write_package_table(table: pd.DataFrame, meta: dict):
    """Write fap_tables.json; methods not simulated keep their existing entries."""
    old = json.loads(PACKAGE_TABLE.read_text()) if PACKAGE_TABLE.exists() else {}
    Ns = sorted(int(n) for n in table["N"].unique())
    doc = {"description": "Per-bin white-noise thresholds for bins of N independent chi^2_2 "
                          "points (scripts/make_fap_tables.py). ACF: thresholds of MCC - mu0, "
                          "with mu0 the white-noise mean MCC. ACV: the same for the mean "
                          "|autocovariance|. CoV: thresholds of the CoV.",
           "N": Ns}
    today = datetime.date.today().isoformat()
    if not old:
        doc.update(created=today, **meta)
    else:                                     # keep the metadata of the existing table
        doc.update({k: v for k, v in old.items() if k not in METHODS and k not in doc})
    for method in METHODS:
        t = table[table["method"] == method].sort_values("N")
        if t.empty:
            if method in old:
                if old["N"] != Ns:
                    raise ValueError(f"N grid differs from the package table; rerun {method} too")
                doc[method] = old[method]
            continue
        doc[method] = {f"{p:g}": [round(v, 8) for v in t[f"p{p:g}"]] for p in PERCENTILES}
        if method in CENTRED:
            doc[method]["mu0"] = [round(v, 8) for v in t["mu0"]]
        doc[f"{method}_meta"] = {"created": today, **meta,
                                 "bins_per_N": t["n_bins"].astype(int).tolist()}
    PACKAGE_TABLE.write_text(json.dumps(doc, indent=1))
    print("wrote", PACKAGE_TABLE)


def violins(out_dir: Path, table: pd.DataFrame):                    # step 5
    """Violin plots of the statistic per N with the two percentiles (fap_violin.png)."""
    import matplotlib.pyplot as plt

    methods = [m for m in METHODS if m in set(table["method"])]
    fig, axs = plt.subplots(len(methods), 1, figsize=(12, 3 + 3 * len(methods)), dpi=150,
                            sharex=True, squeeze=False)
    axs = axs[:, 0]
    for ax, method in zip(axs, methods):
        t = table[table["method"] == method].sort_values("N")
        Ns = t["N"].tolist()
        pos = np.arange(len(Ns))
        data = []
        for N in Ns:
            s = np.load(out_dir / f"statistics_{method}_N{N}.npy", mmap_mode="r")
            step = max(1, s.size // 200_000)                  # subsample for the KDE
            data.append(used_statistic(method, s[::step])[0])
        parts = ax.violinplot(data, pos, widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("lightsteelblue")
            b.set_alpha(0.8)
        for p, c in zip(PERCENTILES, ("tab:orange", "tab:red")):
            ax.plot(pos, t[f"p{p:g}"], "o-", color=c, ms=4, lw=1, label=f"{p:g}th percentile")
        ax.set_ylabel({"ACF": "ACF: MCC - mu0", "ACV": "ACV: mean|ACV| - mu0"}.get(method, "CoV"))
        ax.legend(fontsize=8)
    axs[-1].set_xticks(pos)
    axs[-1].set_xticklabels([str(N) for N in Ns])
    axs[-1].set_xlabel("N (data points per bin)")
    fig.suptitle(f"white noise, >= {table['n_bins'].min():.0e} non-overlapping bins per N")
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
    ap.add_argument("--from-statistics", metavar="DIR",
                    help="no simulation: tables from the statistics_*.npy files in DIR")
    ap.add_argument("--plot-only", metavar="OUT_DIR", help="redraw the violins from OUT_DIR")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS,
                    help="methods to simulate (the others keep their package-table entries)")
    ap.add_argument("--max-points", type=float, default=None,
                    help="at most this many points per N (fewer bins for large N)")
    args = ap.parse_args()

    if args.plot_only:
        out_dir = Path(args.plot_only)
        print("wrote", violins(out_dir, pd.read_csv(out_dir / "fap_thresholds.csv")))
        return

    t0 = time.perf_counter()
    if args.from_statistics:
        out_dir = Path(args.from_statistics)
        Ns = sorted({int(f.stem.split("_N")[1]) for f in out_dir.glob("statistics_*_N*.npy")})
        rows = [summarise(m, N, np.load(out_dir / f"statistics_{m}_N{N}.npy"))
                for N in Ns for m in METHODS if (out_dir / f"statistics_{m}_N{N}.npy").exists()]
        meta = {"source": f"statistics in {out_dir.name}"}
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Ns = [2 ** k for k in range(int(np.log2(args.n_min)), int(np.log2(args.n_max)) + 1)]
        n_bins, pool = int(args.n_bins), int(args.pool_size)
        seeds = np.random.SeedSequence(args.seed).spawn(len(Ns))
        print(f"N = {Ns[0]} ... {Ns[-1]}, {n_bins:.0e} bins per N, "
              f"{n_bins * sum(Ns):.1e} points in total")
        rows = []
        for N, ss in zip(Ns, seeds):
            t = time.perf_counter()
            nb = n_bins if args.max_points is None else max(1000, min(n_bins, int(args.max_points) // N))
            stats = simulate_n(N, nb, pool, np.random.default_rng(ss), args.methods)
            for method, s in stats.items():
                np.save(out_dir / f"statistics_{method}_N{N}.npy", s)
                rows.append(summarise(method, N, s))
            del stats
            print(f"  N = {N:6d}  {time.perf_counter() - t:7.1f} s   "
                  + "   ".join(f"{r['method']} 95/99: {r['p95']:.4f} / {r['p99']:.4f}"
                               for r in rows[-len(args.methods):]), flush=True)
        meta = {"seed": args.seed, "n_bins_per_N": n_bins}

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "fap_thresholds.csv", index=False, float_format="%.8g")
    if not args.no_package:
        write_package_table(table, meta)
    print("wrote", out_dir / "fap_thresholds.csv", violins(out_dir, table),
          f"({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
