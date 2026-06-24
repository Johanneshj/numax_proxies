import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional

def plot(
        frequency : NDArray,
        power : NDArray,
        envelope : NDArray,
        ax : NDArray,
        analytical_signal : Optional[NDArray] = None        
): 
    # Plot original spectrum
    ax.loglog(frequency, power, c="gray", label="spectrum")
    if analytical_signal is not None:
        # Plot analytical signal if provided
        ax.loglog(frequency, analytical_signal, c="k", ls='-', label="analytical signal")
    # Plot envelope
    ax.loglog(frequency, envelope, c='k', label="envelope")
    # Set labels
    ax.set_xlabel("frequency [μHz]")
    ax.set_ylabel("power spectral density")
    # Set x- and y-lims
    ax.set_xlim(np.min(frequency), np.max(frequency))
    # Legend
    ax.legend(loc="upper left")

