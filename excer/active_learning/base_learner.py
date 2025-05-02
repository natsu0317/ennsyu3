class ActiveLearner:
    """Base class for active learning strategies."""
    def __init__(self, model):
        self.model = model
        self.labeled_indices = []
        
    def select_samples(self, X_pool, n_samples=1):
        """Select samples from the pool."""
        pass
    
    def update_model(self, X, y):
        """Update the model with new labeled samples."""
        self.model.fit(X, y)
        
    def add_labeled_indices(self, indices):
        """Add indices to the labeled set."""
        self.labeled_indices.extend(indices)