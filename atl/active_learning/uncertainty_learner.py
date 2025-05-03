import numpy as np
from active_learning.base_learner import ActiveLearner
from utils.losses import entropy

# 不確実性に基づくアクティブラーニング戦略

class UncertaintyActiveLearner(ActiveLearner):
    def __init__(self, model):
        super().__init__(model)

    def select_samples(self, X_pool, n_samples=1, excluded_indices=None):
        # エントロピーに基づいて最も不確実なサンプルを選択
        # excluded_indices: 選択から除外するindexのlist
        if excluded_indices is None:
            excluded_indices = []
        probs = self.model.predict_proba(X_pool) # poolのサンプルの予測確率
        uncertainties = entropy(probs)
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        uncertainties[~mask] = -np.inf
        # entropyが高い=予測が均等=予測が曖昧
        selected_indices = np.argsort(uncertainties)[-n_samples:][:-1]
        return selected_indices
        