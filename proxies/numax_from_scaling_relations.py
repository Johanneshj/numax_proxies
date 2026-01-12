from . import(
    compute_numaxes
)

class NumaxFromScalingRelations:
    def __init__(self, id, *args, **kwargs):
        self._id = id or "unknown"
        
        self._numaxes = None
    
    
    def compute(self, *args, **kwargs):
        """
        Compute νmax using the scaling relations.
        """ 
        self._numaxes = compute_numaxes(id=self._id, *args, **kwargs)
        return self._numaxes
        
