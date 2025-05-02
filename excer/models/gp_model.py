import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from models.base_model import BaseModel

class GPModel(BaseModel):
    """Gaussian Process model for classification."""
    def __init__(self, kernel=None):
        super().__init__()
        if kernel is None:
            kernel = 1.0 * RBF(length_scale=1.0)
        self.model = GaussianProcessClassifier(kernel=kernel)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)