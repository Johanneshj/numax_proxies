import numpy as np

def calculate_noise(flux, noise_std):
    '''
        Add gaussian white noise to flux values

        Inputs:
            flux    : flux array
            std     : magnitude of white noise ppm^2/muHz

        Output:
            noise   : noise same length as flux
    '''
    noise = np.random.normal(1e-10, noise_std, len(flux))
    return noise