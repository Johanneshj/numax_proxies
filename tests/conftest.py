import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# test the source tree, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def kepler_grid():
    """Frequency grid like a 4-year Kepler long-cadence PSD (0.008 - 283 muHz)."""
    return np.arange(1, 35714) * 0.00793
