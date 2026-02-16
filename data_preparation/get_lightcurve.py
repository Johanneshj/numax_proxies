import lightkurve as lk
import glob
import numpy as np


class GetLightcurve:
    def __init__(
        self,
        fits_files_folder=None,
        lc_file=None,
        target=None,
        cadence=None,
        sector=None,
        quarter=None,
        mission=None,
        author=None,
        time=None,
        flux=None,
        flux_err=None,
        *args,
        **kwargs,
    ):

        self._id = target
        self._cadence = cadence
        self._sector = sector
        self._quarter = quarter
        self._mission = mission
        self._author = author

        self._fits_files_folder = fits_files_folder
        self._lc_file = lc_file

        self._time = None
        self._flux = None
        self._flux_err = None

        # Get light curve from lists
        if all(info is not None for info in (time, flux, flux_err)):
            self._time = time
            self._flux = flux
            self._flux_err = flux_err
            self.lightcurve_from_lists()

        # Get light curve from .csv file
        elif lc_file is not None:
            self._lc_file = lc_file
            self.lightcurve_from_file()

        # Get light curve from fits files
        elif fits_files_folder is not None:
            self._fits_files_folder = fits_files_folder
            if "KIC" in self._id.upper():
                self.lightcurve_from_kepler_fits()
            elif "TIC" in self._id.upper():
                self.lightcurve_from_tess_fits()
        
        # Get light curve from LightKurve from target name
        elif target is not None:
            self.set_lightcurve_info(
                author=author,
                cadence=cadence,
                sector=sector,
                quarter=quarter,
                mission=mission,
            )
            self.lightcurve_from_target_name()

        # Default to test star
        else:
            self.template_lightcurve()

    # ----------------------------
    # Light curve options
    # ----------------------------
    def lightcurve_from_kepler_fits(self):
        """Load LC Kepler fits files"""
        # Only works for kepler right now
        n = int("".join(filter(str.isdigit, self._id)))
        target = f"{int(n):09d}"
        fits_files_folder = self._fits_files_folder
        lc_files = glob.glob(f"{fits_files_folder}/kplr{target}*")
        if len(lc_files) > 0:
            lcs = [lk.read(file, quality_bitmask="default") for file in lc_files]
            lccoll = lk.LightCurveCollection(lcs)
            lc = lccoll.stitch().remove_nans().remove_outliers(5)
            self._time = lc.time.value
            self._flux = lc.flux.value
            self._flux_err = lc.flux_err.value
            return self
        else:
            raise FileNotFoundError(f"No fits files for target {self._id}")

    def lightcurve_from_tess_fits(self):
        """Load LC TESS fits files"""
        # Only works for kepler right now
        n = int("".join(filter(str.isdigit, self._id)))
        target = f"{int(n):016d}"
        fits_files_folder = self._fits_files_folder
        lc_files = glob.glob(f"{fits_files_folder}/*{target}*")
        if len(lc_files) > 0:
            lcs = [lk.read(file, quality_bitmask="default") for file in lc_files]
            lccoll = lk.LightCurveCollection(lcs)
            lc = lccoll.stitch().remove_nans().remove_outliers(5)
            self._time = lc.time.value
            self._flux = lc.flux.value
            self._flux_err = lc.flux_err.value
            return self
        else:
            raise FileNotFoundError(f"No fits files for target {self._id}")

    def lightcurve_from_file(self):
        """Load lc from .csv file"""
        data = np.genfromtxt(
            self._lc_file, delimiter=",", names=["time", "flux", "flux_err"]
        )
        mask = (
            ~np.isnan(data["time"])
            & ~np.isnan(data["flux"])
            & ~np.isnan(data["flux_err"])
        )
        data = data[mask]
        self._time = np.array(data["time"])
        self._flux = np.array(data["flux"])
        self._flux_err = np.array(data["flux_err"])
        return self

    def lightcurve_from_target_name(self):
        """Use LightKurve to grab lc from id"""
        if "KIC" in self._id.upper():
            search_results = lk.search_lightcurve(
                target=self._id,
                author=self._author,
                cadence=self._cadence,
                quarter=self._quarter,
            )
        elif "TIC" in self._id.upper():
            search_results = lk.search_lightcurve(
                target=self._id,
                author=self._author,
                cadence=self._cadence,
                sector=self._sector,
            )
        lc = (
            search_results.download_all(quality_bitmask="default")
            .stitch()
            .remove_nans()
            .remove_outliers(5)
        )
        self._time = lc.time.value
        self._flux = lc.flux.value
        self._flux_err = lc.flux_err.value
        return self

    def lightcurve_from_lists(self):
        lc = lk.LightCurve(
            time=self._time,
            flux=self._flux,
            flux_err=self._flux_err
        )
        lc = (
            lc
            .stitch()
            .remove_nans()
            .remove_outliers(5)
        )
        self._time = lc.time.value
        self._flux = lc.flux.value
        self._flux_err = lc.flux_err.value
        return self

    def template_lightcurve(self):
        """Default to KIC 12008916 if nothing else specified"""
        search_results = lk.search_lightcurve(
            "KIC12008916",
            mission="Kepler",
            cadence="long",
            author="Kepler",
            quarter=np.arange(0, 60),
        )
        self._id = "KIC12008916"
        lc = search_results.download_all().stitch()
        self._time = lc.time.value
        self._flux = lc.flux.value
        self._flux_err = lc.flux_err.value
        return self

    # ----------------------------
    # Utilities
    # ----------------------------
    def set_lightcurve_info(self, cadence, quarter, sector, mission, author):
        self._cadence = cadence
        self._quarter = quarter
        self._sector = sector
        if mission is None:
            self._mission = "Kepler" if "KIC" in self._id.upper() else "TESS"
        else:
            self._mission = mission

        if author is None:
            self._author = "Kepler" if "KIC" in self._id.upper() else "SPOC"
        else:
            self._author = author

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def final_lc(self):
        return self._time, self._flux, self._flux_err
