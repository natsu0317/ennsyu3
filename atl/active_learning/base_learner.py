class ActiveLearner:
    def __init__(self, model):
        self.model = model
        self.labeled_indices = []
    
    def select_samples(self, X_pool, n_samples=1):
        # return 選択されたサンプルのindex list
        pass

    def update_model(self, X, y):
        # 新しいラベル付きサンプルでモデル更新
        self.model.fit(X, y)

    def add_labeled_indices(self, indices):
        self.labeled_indices.extend(indices)