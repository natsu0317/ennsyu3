class BaseModel:
    # モデルの初期化
    def __init__(self):
        pass

    # データからモデルの学習
    def fit(self, X, y):
        # X: numpy.ndarray(特徴量データ)
        # y: numpy.ndarray(ラベルデータ)
        pass

    # クラスラベルの予測
    def predict(self, X):
        pass

    # クラス確率の予測
    def predict_proba(self, X):
        pass
