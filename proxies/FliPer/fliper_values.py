from .FLIPER import DATA_PREPARATION
from .FLIPER import FLIPER
import numpy as np

def calculate_FliPer_values(lc, star_tab_psd_80, star_tab_psd_20, noise):
    """
        Calculate FliPer values to be used later for numax estimate.
        FliPer values are, for a given frequency range, the averaged PSD power minus the photon noise.
        Fp = <PSD> - Pn.
        We utilize routines already available in FliPer (Bugnet et al. 2018).

        Input:
            Filtered PSDs (20d and 80d filter)
        
        Output:
            Fp02 :: Fp from 0.2 muHz
            Fp07 :: Fp from 0.7 muHz
            Fp7 :: Fp from 7 muHz
            Fp20 :: Fp from 20 muHz
            Fp50 :: Fp from 50 muHz
    """
    fliper_20 = Fp_20_days(star_tab_psd_20, noise)
    fliper_80 = Fp_80_days(star_tab_psd_80, noise)
    cadence = np.mean(np.diff(lc.time.value * 86400))
    return (
        fliper_80["Fp02"],
        fliper_20["Fp07"],
        fliper_20["Fp7"],
        fliper_20["Fp20"],
        fliper_20["Fp50"],
        cadence,
    )

def Fp_20_days(star_tab_psd_20, noise):
        """
        Compute FliPer value from 0.7, 7, 20, and 50 muHz to Nyquist with 20 days filtered data.
        Stolen from Bugnet et al. (2018)!
        We do not use Kepler or TESS mags to estimate noise, instead we use the last 100 bins of the PSD.
        """
        star_tab_psd_20 =   DATA_PREPARATION().APODIZATION(star_tab_psd_20)
        end_20          =   (np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_20)[0]*1e6))
        # noise           =   DATA_PREPARATION().MAG_COR_KEP(star_tab_psd_20, kepmag)
        Fp07            =   DATA_PREPARATION().REGION(star_tab_psd_20, 0.7, end_20) - noise
        Fp7             =   DATA_PREPARATION().REGION(star_tab_psd_20, 7, end_20) - noise
        Fp20            =   DATA_PREPARATION().REGION(star_tab_psd_20, 20, end_20) - noise
        Fp50            =   DATA_PREPARATION().REGION(star_tab_psd_20, 50, end_20) - noise
        sig_Fp07        =   Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 0.7, end_20))
        sig_Fp7         =   Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 7, end_20))
        sig_Fp20        =   Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 20, end_20))
        sig_Fp50        =   Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_20, 50, end_20))

        return {
            "Fp07": Fp07, "Fp7": Fp7, "Fp20": Fp20, "Fp50": Fp50,
            "sig_Fp07": sig_Fp07, "sig_Fp7": sig_Fp7,
            "sig_Fp20": sig_Fp20, "sig_Fp50": sig_Fp50,
        }

def Fp_80_days(star_tab_psd_80, noise):
    star_tab_psd_80 = DATA_PREPARATION().APODIZATION(star_tab_psd_80)
    end_80 = np.amax(DATA_PREPARATION().GET_ARRAY(star_tab_psd_80)[0] * 1e6)

    Fp02 = DATA_PREPARATION().REGION(star_tab_psd_80, 0.2, end_80) - noise
    sig_Fp02 = Fp_error(DATA_PREPARATION().CUT_SPECTRA(star_tab_psd_80, 0.2, end_80))

    return {"Fp02": Fp02, "sig_Fp02": sig_Fp02}

def Fp_error(power): #GUY
        """
        Compute errors on FliPer values du to noise.
        """
        n           =   50                                                                      #   rebin of the spectra to have normal distribution on the uncertainties
        Ptmp        =   np.array([np.sum(power[i*n:(i+1)*n]) for i in range(int(len(power)/n))])#   power on the rebin
        Ptot        =   np.sum(Ptmp)                                                            #   total power
        sig_Ptot    =   (np.sum((2 * Ptmp / 2 / n * n**0.5)**2))**0.5                           #   uncertainties on total power
        error_Fp    =   ((sig_Ptot / len(power))**2)**0.5
        return error_Fp