"""Entropy check: white noise passes rarely, a synthetic comb is found at Delta-nu."""
import numpy as np

from numax_proxies.comb_scan import relative_power
from numax_proxies.entropy_check import VERSIONS, entropy_check
from numax_proxies.seismic import delta_nu
from numax_proxies.synthetic import synthetic_psd, white_noise_psd

GRID = np.arange(0.0079, 283.2, 0.0079)          # Kepler LC, 4 years


def check(psd, numax, **kw):
    return entropy_check(psd.frequency, relative_power(psd.frequency, psd.psd), numax, **kw)


def test_white_noise_false_alarms():
    rng = np.random.default_rng(3)
    grid = GRID[(GRID > 50) & (GRID < 150)]
    hits = {v: 0 for v in VERSIONS}
    for _ in range(100):
        r = check(white_noise_psd(grid, rng), 100.0, aim_at="none")
        for v in VERSIONS:
            hits[v] += r[v]["significant"]
    assert all(h <= 5 for h in hits.values()), hits          # 1 % expected


def test_finds_delta_nu():
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    r = check(psd, 90.0)
    for v in VERSIONS:
        assert r[v]["significant"], v
        assert abs(r[v]["D"] / float(delta_nu(100.0)) - 1) < 0.03, (v, r[v]["D"])


def test_entropy_check_in_pipeline():
    from numax_proxies import NumaxProxies
    psd = synthetic_psd(100.0, 3.0, frequency=GRID, rng=np.random.default_rng(1))
    p = NumaxProxies.from_psd(psd.frequency, psd.psd, {
        "CONFIG": {"entropy_check": True, "save_results": False},
        "COV_CONFIG": {"plot": False, "save_info": False, "save_results": False}})
    p.compute_numax_from_CoV()
    t = p.entropy_check_table
    power = t[t.version == "power"]
    assert len(power) == 11 and len(t[t.version == "peaks"]) == 77    # 11 bins (x 7 peak numbers)
    assert set(t[t.version == "power"].width) == {8.0} and set(t[t.version == "peaks"].width) == {6.0}
    assert power.numax.notna().all()
    assert abs(p.numax_estimates["numax_Hpower_CoV"].n / 100.0 - 1) < 0.15
    assert "dnu_Hpower_CoV" in p.dnu_estimates


def test_null_table_matches_simulation():
    """The shipped table agrees with a fresh simulation (1 % threshold within 10 %)."""
    from numax_proxies import entropy_check as ec
    tab = ec._from_table("power|10|8")
    sim = ec._sim_power(10, 8.0, 1000, seed=7)
    assert tab is not None
    assert abs(ec.threshold(tab, 0.01) / ec.threshold(sim, 0.01) - 1) < 0.1
