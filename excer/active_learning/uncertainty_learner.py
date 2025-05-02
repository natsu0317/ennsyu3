import numpy as np
from active_learning.base_learner import ActiveLearner
from utils.losses import entropy

class UncertaintyActiveLearner(ActiveLearner):
    """Active learner using uncertainty sampling."""
    def __init__(self, model):
        super().__init__(model)
        
    def select_samples(self, X_pool, n_samples=1, excluded_indices=None):
        """Select samples based on entropy uncertainty."""
        if excluded_indices is None:
            excluded_indices = []
            
        # Get predictions for the pool
        probs = self.model.predict_proba(X_pool)
        
        # Calculate entropy
        uncertainties = entropy(probs)
        
        # Mask out already labeled samples
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        uncertainties[~mask] = -np.inf
        
        # Select the most uncertain samples
        selected_indices = np.argsort(uncertainties)[-n_samples:][::-1]
        
        return selected_indices