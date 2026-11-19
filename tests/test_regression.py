"""Results must not change unintentionally.

Reference: tests/data/reference.npz, from a fixed-seed synthetic spectrum
(tests/make_reference.py). Regenerate it only for intended changes.
"""
import numpy as np
import pytest

from make_reference import OUT, compute


@pytest.fixture(scope="module")
def results():
    return compute(), np.load(OUT)


@pytest.mark.parametrize("method", ["ACF", "CoV"])
def test_matches_reference(results, method):
    new, ref = results
    np.testing.assert_array_equal(new[f"{method}_keys"], ref[f"{method}_keys"])
    np.testing.assert_allclose(new[f"{method}_curve"], ref[f"{method}_curve"], rtol=1e-10)
    np.testing.assert_allclose(new[f"{method}_numax"], ref[f"{method}_numax"], rtol=1e-6)
