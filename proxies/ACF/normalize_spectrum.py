from scipy.ndimage import median_filter
import numpy as np
from numpy.typing import NDArray

def calculate_relative_power(frequency : NDArray, power : NDArray):
    """Subtract and normalize PSD by median filter (Viani+ 2019)"""

    if np.max(frequency) > 300:
        ws = 100  # muHz
    else:
        ws = 10  # muHz

    df = np.median(np.diff(frequency))
    wp = int(ws / df)

    med_filter = median_filter(power, size=wp, mode="reflect")
    rel_power = (power - med_filter) / med_filter

    return rel_power, med_filter