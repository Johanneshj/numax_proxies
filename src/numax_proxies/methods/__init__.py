"""numax methods: ACF, CoV and scaling relations."""
from .acf import NumaxFromACF
from .cov import NumaxFromCoV
from .scaling import numax_from_scaling_relations

__all__ = ["NumaxFromACF", "NumaxFromCoV", "numax_from_scaling_relations"]
