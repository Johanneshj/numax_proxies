import numpy as np
import matplotlib.pyplot as plt

def batch_fft_acf(power_windows):
    """
    Autocorrelation by means of FFT.
    Suitable for SC data if sliding_window is linear.

    Input:
        power_windows :: list of binned windows (sliding)

    Output:
        abs(acf) :: absolute 2D autocorrelation function
    """

    _, n_points = power_windows.shape
    nfft = 1 << (2 * n_points - 1).bit_length()
    power_windows -= np.mean(power_windows, axis=1, keepdims=True)

    fft = np.fft.rfft(power_windows, n=nfft, axis=1)
    psd = np.abs(fft) ** 2
    acf = np.fft.irfft(psd, n=nfft, axis=1)
    acf = acf[:, :n_points]
    acf /= np.max(acf, axis=1, keepdims=True)

    return abs(acf)


def abs_acf(x):
    """
    Autocorrelation by means of np.correlate(x,x).
    Normalization tailored to log sliding windows

    Input:
        x :: PSD values

    Output:
        np.abs(corr) * scaling :: normalized absolute autocorrelation
    """

    # Subtract mean
    x -= np.mean(x)  

    # Perform ACF on segment (x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]  # grab only the positive lags
    
    scaling = np.var(x) / np.sqrt(len(x))

    return np.abs(corr) * scaling

def abs_acf_linear(x):
    """
    Autocorrelation by means of np.correlate(x,x).
    Normalization is different when we are using a linear sliding window.

    Input:
        x :: PSD values

    Output:
        np.abs(corr / np.max(corr)) :: normalized absolute autocorrelation
    """

    # Subtract mean
    x -= np.mean(x)  

    # Perform ACF on segment (x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]  # grab only the positive lags

    return np.abs(corr / np.max(corr))


