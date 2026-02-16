"""
Import from all subfolders everything we'll need
"""

# from . import ACF, ScalingRelations, CoV, FliPer

# __all__ = [*ACF.__all__, *ScalingRelations.__all__, *CoV.__all__, *FliPer.__all__]

from .numax_from_ACF import NumaxFromACF
from .numax_from_coefficients_of_variation import NumaxFromCoefficientsOfVariation
from .numax_from_FliPer import NumaxFromFliPer
from .numax_from_scaling_relations import NumaxFromScalingRelations

__all__ = [
    "NumaxFromACF",
    "NumaxFromCoefficientsOfVariation",
    "NumaxFromFliPer",
    "NumaxFromScalingRelations"
]

# from .ACF import *
# from .ScalingRelations import *
# from .CoV import *
# from .FliPer import *
