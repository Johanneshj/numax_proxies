from .ScalingRelations import (
    compute_numaxes
)


class NumaxFromScalingRelations:
    def __init__(self, id, *args, **kwargs):
        self._id = id or "unknown"

        if kwargs.get("gaia_query_dict"):
            self._gaia_query_dict = kwargs.get("gaia_query_dict")
        else:
            self._gaia_query_dict = None

        self._numaxes = None

    def compute(self, *args, **kwargs):
        """
        Compute νmax using the scaling relations.
        """
        self._numaxes = compute_numaxes(
            id=self._id, gaia_query_dict=self._gaia_query_dict, *args, **kwargs
        )
        return self._numaxes
