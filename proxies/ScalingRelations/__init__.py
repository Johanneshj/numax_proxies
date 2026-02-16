"""
Functions for scaling relations method
"""

from .query import (
    query_gaia, 
    get_query, 
    return_dict
)
from .scaling_relations import (
    numax_scaling_relations,
    make_broadcastable_uarray,
    compute_numaxes,
)

__all__ = [
    "query_gaia",
    "get_query",
    "numax_scaling_relations",
    "make_broadcastable_uarray",
    "compute_numaxes",
    "return_dict",
]
