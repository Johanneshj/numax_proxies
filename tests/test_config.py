import warnings

import pytest

from numax_proxies.config import config_from_dict, load_config

from conftest import EXAMPLES


def test_example_config_loads():
    cfg = load_config(EXAMPLES / "KIC1872517.yaml")
    assert cfg.star.target == "KIC1872517"
    assert cfg.lightcurve.lc_file.endswith("LC_KIC1872517.parquet")   # resolved next to the YAML
    assert cfg.acf_config.width_factor == [1.0, 5.0, 15.0]


def test_empty_config_uses_defaults():
    cfg = config_from_dict({})
    assert cfg.acf_config.min_freq == 1.0
    assert cfg.config.random_seed == 42


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown key"):
        config_from_dict({"ACF_CONFIG": {"widht_factor": [1]}})


def test_unknown_section_raises():
    with pytest.raises(ValueError, match="Unknown section"):
        config_from_dict({"ACF": {}})


def test_deprecated_keys_warn_but_load():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config_from_dict({"COV_CONFIG": {"use_linear_bins": False}, "EACF_CONFIG": {"plot": False}})
    assert any("use_linear_bins" in str(x.message) for x in w)


def test_scalar_grid_values_become_lists():
    cfg = config_from_dict({"COV_CONFIG": {"width_factor": 3}})
    assert cfg.cov_config.width_factor == [3.0]


@pytest.mark.parametrize("section,key,value", [
    ("COV_CONFIG", "FAP_threshold", "90p0"),
    ("ACF_CONFIG", "width_factor", [0, 1]),
    ("CONFIG", "avg_psd_method", "mode"),
    ("CONFIG", "avg_psd_overlap", 1.0),
])
def test_invalid_values_raise(section, key, value):
    with pytest.raises(ValueError):
        config_from_dict({section: {key: value}})


def test_star_parameter_needs_error():
    with pytest.raises(ValueError, match="teff_err"):
        config_from_dict({"STAR": {"teff": [4800]}})
