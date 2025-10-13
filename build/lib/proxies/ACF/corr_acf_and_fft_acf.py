import numpy as np

def batch_fft_acf(power_windows):
    '''
        Autocorrelation by means of FFT.
        Suitable for SC data.

        Input:
            power_windows :: list of binned windows (sliding)
        
        Output:
            abs(acf) :: absolute 2D autocorrelation function
    '''

    _, n_points = power_windows.shape
    nfft = 1 << (2 * n_points - 1).bit_length()
    power_windows -= np.mean(power_windows, axis=1, keepdims=True)

    # start = time.time()
    fft = np.fft.rfft(power_windows, n=nfft, axis=1)
    # end = time.time()
    # print(f"FFT computation time: {end - start:.4f} seconds")

    # start = time.time()
    psd = np.abs(fft)**2
    # end = time.time()
    # print(f"PSD computation time: {end - start:.4f} seconds")

    # start = time.time()
    acf = np.fft.irfft(psd, n=nfft, axis=1)
    # end = time.time()
    # print(f"Inverse FFT computation time: {end - start:.4f} seconds")

    # start = time.time()
    acf = acf[:, :n_points]
    # end = time.time()
    # print(f"Lags time: {end - start:.4f} seconds")

    # start = time.time()
    acf /= np.max(acf, axis=1, keepdims=True)
    # end = time.time()
    # print(f"Norm time: {end - start:.4f} seconds")

    return abs(acf)

def abs_acf(x):
    '''
        Autocorrelation by means of np.correlate(x,x).
        Suitable for LC data.
        Calculated on a row-by-row basis.

        Input:
            x :: PSD values

        Output:
            np.abs(corr / np.max(corr)) :: normalized absolute autocorrelation
    '''
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:] # grab only the positive lags
    return np.abs(corr / np.max(corr))