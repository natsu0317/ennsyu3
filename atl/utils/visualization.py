import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title, labeled_indices=None, test_indices=None, feedback_indices=None):
    """
    モデルの決定境界を可視化します
    
    Parameters:
    -----------
    model : BaseModel
        可視化するモデル
    X : numpy.ndarray
        特徴量データ
    y : numpy.ndarray
        ラベルデータ
    title : str
        プロットのタイトル
    labeled_indices : list, optional
        ラベル付きデータのインデックス
    test_indices : list, optional
        テストデータのインデックス
    feedback_indices : list, optional
        フィードバックデータのインデックス
    
    Returns:
    --------
    plt : matplotlib.pyplot
        プロットオブジェクト
    """
    plt.figure(figsize=(10, 8))
    
    # グリッドを定義
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # グリッド上で予測
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    
    if Z.shape[1] > 1:  # 多クラス
        Z = Z[:, 1]
    
    Z = Z.reshape(xx.shape)
    
    # カラーマップを設定（より鮮明なコントラスト）
    cmap = plt.cm.RdBu_r
    
    # 決定境界をプロット（コントラストを高める）
    plt.contourf(xx, yy, Z, alpha=1.0, cmap=cmap, levels=np.linspace(0, 1, 11))
    
    # 決定境界線を太く、はっきりと表示
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-', linewidths=2)
    
    # データポイントをプロット（エッジを強調）
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidths=1, cmap=cmap, alpha=0.8)
    
    # ラベル付き、テスト、フィードバックポイントをハイライト
    if labeled_indices is not None:
        plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], s=100, facecolors='none', edgecolors='g', linewidths=2, label='AL Sample')
    
    if test_indices is not None:
        plt.scatter(X[test_indices, 0], X[test_indices, 1], s=100, facecolors='none', edgecolors='b', linewidths=2, label='AT Sample')
    
    if feedback_indices is not None:
        plt.scatter(X[feedback_indices, 0], X[feedback_indices, 1], s=100, facecolors='none', edgecolors='r', linewidths=2, label='Feedback Sample')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # 軸の範囲を設定
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # グリッド線を追加
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return plt