"""White-noise null table of the entropy check (src/numax_proxies/entropy_null.json).

For every setting (window width, bins, and for the peaks version the number of peaks and their
minimum separation): contrast thresholds at the false-alarm levels PROBS and the mean G, from
--sims white-noise simulations. Settings outside the table are simulated at run time.

    python scripts/make_entropy_nulls.py [--sims 10000] [--processes 30] [--widths 8]
        [--bins 4 20] [--peaks 5 100] [--kinds power peaks] [--merge]
        (--merge: keep the other entries of the table)
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
OUT = SRC / "numax_proxies" / "entropy_null.json"
WIDTHS = (8.0,)
BINS = range(4, 21)
PEAKS = range(5, 101)
SEPS = (0.05, 0.125)


def job(args):
    import sys
    sys.path.insert(0, str(SRC))
    from numax_proxies import entropy_check as ec
    key, kind, nb, width, n, sep, sims = args
    r = (ec._sim_power(nb, width, sims) if kind == "power"
         else ec._sim_peaks(nb, width, n, sep, sims))
    return key, dict(q=[None if v == float("inf") else round(float(v), 5) for v in r["q"]],
                     mean_g=round(r["mean_g"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--processes", type=int, default=30)
    ap.add_argument("--widths", type=float, nargs="+", default=list(WIDTHS))
    ap.add_argument("--bins", type=int, nargs=2, default=[BINS.start, BINS.stop - 1])
    ap.add_argument("--peaks", type=int, nargs=2, default=[PEAKS.start, PEAKS.stop - 1])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["power", "peaks"])
    a = ap.parse_args()
    bins, peaks = range(a.bins[0], a.bins[1] + 1), range(a.peaks[0], a.peaks[1] + 1)
    jobs = [(f"power|{nb}|{w:g}", "power", nb, w, 0, 0.0, a.sims) for w in a.widths for nb in bins
            if "power" in a.kinds]
    jobs += [(f"peaks|{nb}|{w:g}|{n}|{s:g}", "peaks", nb, w, n, s, a.sims)
             for w in a.widths for nb in bins for n in peaks for s in SEPS if "peaks" in a.kinds]
    jobs.sort(key=lambda j: -(j[2] * (40 if j[1] == "power" else j[4])))   # slow ones first
    with mp.get_context("spawn").Pool(a.processes) as pool:
        table = dict(pool.imap_unordered(job, jobs))
    import sys
    sys.path.insert(0, str(SRC))
    from numax_proxies import entropy_check as ec
    meta = dict(d_range=list(ec.D_RANGE), plot_range=list(ec.PLOT_RANGE),
                oversample=ec.OVERSAMPLE, probs=len(ec.PROBS))
    if a.merge and OUT.exists():
        old = json.loads(OUT.read_text())
        assert old["meta"] == meta, "table made with other settings"
        table = {**{k: v for k, v in old.items() if "|" in k}, **table}
    out = dict(meta=meta, sims=a.sims, probs=[float(p) for p in ec.PROBS],
               **dict(sorted(table.items())))
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(table)} settings -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
