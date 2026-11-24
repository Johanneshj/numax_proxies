"""numax methods: ACF, ACV, CoV and scaling relations."""
from .acf import NumaxFromACF, NumaxFromACV
from .cov import NumaxFromCoV
from .scaling import numax_from_scaling_relations

__all__ = ["NumaxFromACF", "NumaxFromACV", "NumaxFromCoV", "numax_from_scaling_relations"]
