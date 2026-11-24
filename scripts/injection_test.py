"""Injection test: run the ACF and CoV on synthetic spectra with a known numax (and on pure noise).

    python scripts/injection_test.py --out injection.parquet --n-real 20
    python scripts/injection_test.py --settings my_grid.yaml     # ACF/COV sections only

Prints, per method and width_factor, the detection rate, the fraction within 10 % and the
median absolute error.
"""
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numax_proxies import NumaxProxies                      # noqa: E402
from numax_proxies.synthetic import kepler_lc_grid, synthetic_psd, white_noise_psd  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_GRID = {
    "overlap_scale": [5], "width_factor": [1, 2, 5, 10, 15], "smoothing_factor": [1.0],
    "min_freq": 1.0, "FAP_threshold": "99p0",
}


def run_one(psd, settings):
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, settings, target="synthetic")
    p.compute_numax_from_acf()
    p.compute_numax_from_CoV()
    rows = []
    for name, method in (("ACF", p.acf), ("CoV", p.cov)):
        for (o, w, s), d in method.results.items():
            rows.append(dict(method=name, overlap_scale=o, width_factor=w, smoothing_factor=s,
                             numax=d["fit_numax"].n, numax_err=d["fit_numax"].s))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numax", type=float, nargs="+", default=[15, 30, 60, 120, 200])
    ap.add_argument("--height-ratio", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--n-real", type=int, default=10)
    ap.add_argument("--n-noise", type=int, default=None, help="white-noise spectra (default n-real)")
    ap.add_argument("--settings", help="YAML with ACF_CONFIG/COV_CONFIG/CONFIG sections")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="injection_results.parquet")
    args = ap.parse_args()

    settings = {"ACF_CONFIG": dict(DEFAULT_GRID), "COV_CONFIG": dict(DEFAULT_GRID)}
    if args.settings:
        settings.update(yaml.safe_load(Path(args.settings).read_text()))
    rng = np.random.default_rng(args.seed)
    nu = kepler_lc_grid()

    rows, t0 = [], time.time()
    for numax in args.numax:
        for hr in args.height_ratio:
            for r in range(args.n_real):
                for row in run_one(synthetic_psd(numax, hr, nu, rng=rng), settings):
                    rows.append(dict(row, true_numax=numax, height_ratio=hr, realisation=r))
        logging.warning("numax %g done (%.0f s)", numax, time.time() - t0)
    for r in range(args.n_noise if args.n_noise is not None else args.n_real):
        for row in run_one(white_noise_psd(nu, rng), settings):
            rows.append(dict(row, true_numax=np.nan, height_ratio=0.0, realisation=r))

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}\n")
    print(summarise(df).to_string())


def summarise(df):
    df = df.copy()
    df["detected"] = np.isfinite(df.numax)
    rel = np.abs(df.numax - df.true_numax) / df.true_numax * 100
    df["within10"] = rel <= 10
    df["ape"] = rel
    keys = ["method", "width_factor", "height_ratio"]
    out = df.groupby(keys).agg(detected=("detected", "mean"), within10=("within10", "mean"),
                               medape=("ape", "median"))
    out[["detected", "within10"]] *= 100
    return out.round(1)


if __name__ == "__main__":
    main()
