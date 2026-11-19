"""Regenerate tests/data/reference.npz. Run ONLY when results are meant to change.

    python tests/make_reference.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numax_proxies import NumaxProxies                      # noqa: E402
from numax_proxies.synthetic import kepler_lc_grid, synthetic_psd  # noqa: E402

GRID = {"overlap_scale": [1, 5], "width_factor": [1, 5, 15], "smoothing_factor": [0.5, 1.5],
        "FAP_threshold": "99p0"}
SETTINGS = {"ACF_CONFIG": GRID, "COV_CONFIG": GRID}
OUT = Path(__file__).parent / "data" / "reference.npz"


def compute():
    psd = synthetic_psd(40.0, 3.0, kepler_lc_grid(), rng=np.random.default_rng(2026))
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, SETTINGS, target="reference")
    p.compute_numax_from_acf()
    p.compute_numax_from_CoV()
    out = {}
    for name, m in (("ACF", p.acf), ("CoV", p.cov)):
        keys = sorted(m.results)
        out[f"{name}_keys"] = np.array(keys)
        out[f"{name}_numax"] = np.array([m.results[k]["fit_numax"].n for k in keys])
        curve = "cacf_smoothed" if name == "ACF" else "cov_smoothed"
        out[f"{name}_curve"] = np.concatenate([m.results[k][curve] for k in keys])
    return out


if __name__ == "__main__":
    np.savez(OUT, **compute())
    print("wrote", OUT)
