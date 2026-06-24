from .ScalingRelations import (
    compute_numaxes
)
from typing import Optional
from ..data_preparation.dataclasses import GaiaData, ProcessingConfig, StarInfo

class NumaxFromScalingRelations:
    def __init__(self,
                 star : StarInfo,
                 config : ProcessingConfig,
                 gaia_data : Optional[GaiaData]

        ):
        # Initialize
        self.star = star
        self.config = config
    
        # Is gaia data already found?
        if gaia_data.has_data():
            self.gaia_data = gaia_data
        else:
            self.gaia_data = None

    def compute(self):
        """
        Compute νmax using the scaling relations.
        """
        self.numaxes = compute_numaxes(
            star = self.star, 
            gaia_data = self.gaia_data
        )
        return self

    @property
    def numax_estimates(self) -> dict:
            return self.numaxes
