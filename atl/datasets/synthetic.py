# 図1(a)の合成データセット生成
import numpy as np
from sklearn.datasets import make_moons

def create_moon_dataset_with_clusters():
    # main: moon形状 + sub: moon形状 * 2
    X_main, y_main = make_moons(n_samples=2000, noise=0.15, random_state=42)

    X_cluster1, y_cluster1 = make_moons(n_samples=250, noise=0.08, random_state=42)
    X_cluster1 = X_cluster1 * 0.5 - np.array([2.0, 2.0]) # スケールと位置の調整
    y_cluster1 = 1 - y_cluster1 # ラベル反転

    X_cluster2, y_cluster2 = make_moons(n_samples=250, noise=0.08, random_state=42)
    X_cluster2 = X_cluster2 * 0.5 - np.array([2.0, 2.0]) # スケールと位置の調整
    y_cluster2 = 1 - y_cluster2 # ラベル反転

    X = np.vstack([X_main, X_cluster1, X_cluster2])
    y = np.hstack([y_main, y_cluster1, y_cluster2])

    # データセットをシャッフル
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X,y