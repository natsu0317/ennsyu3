import numpy as np
import matplotlib.pyplot as plt
import types
from sklearn.model_selection import train_test_split
from models.gp_model import GPModel
from datasets.synthetic import create_moon_dataset_with_clusters
from atl_framework.atl import ATLFramework
from utils.visualization import plot_decision_boundary
from utils.losses import cross_entropy_loss
from sklearn.metrics import accuracy_score

def run_synthetic_comparison_experiment():
    """
    合成データを使用して複数の手法（AL sampling, Random, ATL-NF, ATL）を比較します。
    論文の図3に示されている実験を再現します。
    
    Returns:
    --------
    results : dict
        各手法の実験結果
    """
    # データセットを生成
    X, y = create_moon_dataset_with_clusters()
    
    # 初期ラベル付きセットとプールに分割
    X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)
    initial_indices = list(range(len(X_initial)))
    
    print(f"Initial labeled set size: {len(initial_indices)}")
    print(f"Pool size: {len(X_pool)}")
    
    # 比較する手法
    methods = ["AL_sampling", "Random", "ATL-NF", "ATL"]
    
    # 結果を保存する辞書
    results = {}
    
    for method in methods:
        print(f"Running {method}...")
        
        # モデルを初期化
        model = GPModel()
        
        # ATLフレームワークを初期化
        atl = ATLFramework(
            model=model,
            X_pool=X,
            y_pool=y,
            initial_labeled_indices=initial_indices.copy(),
            test_frequency=1,
            test_batch_size=10,
            feedback_ratio=0.5 if method == "ATL" else 0.0,  # ATLのみフィードバックを有効化
            window_size=3
        )
        
        # テスト提案分布の計算方法を手法に応じて修正
        if method == "AL_sampling":
            # ALサンプリングを使用したテスト提案
            def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                if excluded_indices is None:
                    excluded_indices = []
                
                # プールのサンプルに対する予測確率を取得
                probs = self.model.predict_proba(X_pool)
                
                # エントロピーを計算して不確実性を評価
                from utils.losses import entropy
                uncertainties = entropy(probs)
                
                # マスクを作成
                mask = np.ones(len(X_pool), dtype=bool)
                mask[excluded_indices] = False
                uncertainties[~mask] = -np.inf
                
                # 不確実性に基づく提案分布を作成
                q_star = np.zeros(len(X_pool))
                if np.sum(mask) > 0:
                    # 上位の不確実なサンプルに高い確率を割り当て
                    top_indices = np.argsort(uncertainties)[-100:][::-1]
                    q_star[top_indices] = uncertainties[top_indices]
                    q_star = q_star / np.sum(q_star)
                
                return q_star
            
            atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
            
        elif method == "Random":
            # ランダムサンプリングを使用したテスト提案
            def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                if excluded_indices is None:
                    excluded_indices = []
                
                # マスクを作成
                mask = np.ones(len(X_pool), dtype=bool)
                mask[excluded_indices] = False
                
                # 一様分布を作成
                q_star = np.zeros(len(X_pool))
                q_star[mask] = 1
                if np.sum(mask) > 0:
                    q_star = q_star / np.sum(q_star)
                
                return q_star
            
            atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
        
        # 早期停止を無効化
        original_check_early_stopping = atl.check_early_stopping
        atl.check_early_stopping = lambda: False
        
        try:
            # アクティブラーニングを実行
            atl.run_active_learning(n_rounds=20, n_samples_per_round=10)
        finally:
            # 元のメソッドを復元
            atl.check_early_stopping = original_check_early_stopping
        
        # 実際に実行されたラウンド数を取得
        actual_rounds = len(atl.integrated_risk_history)
        print(f"{method} - Actual rounds completed: {actual_rounds}")
        
        # クイズ9とクイズ18の結果を保存
        if actual_rounds >= 9:
            # クイズ9の結果
            quiz_9_indices = atl.test_indices[:min(90, len(atl.test_indices))]
            feedback_9_indices = atl.feedback_indices[:min(45, len(atl.feedback_indices))] if method == "ATL" else []
            
            if len(atl.integrated_risk_history) > 8 and len(atl.true_risk_history) > 8:
                estimation_error_9 = abs(atl.integrated_risk_history[8] - atl.true_risk_history[8])
            else:
                estimation_error_9 = 0
                
            quiz_9_result = {
                'model': model,
                'labeled_indices': atl.labeled_indices[:min(90, len(atl.labeled_indices))],
                'test_indices': quiz_9_indices,
                'feedback_indices': feedback_9_indices,
                'estimation_error': estimation_error_9
            }
        else:
            quiz_9_result = None
            
        if actual_rounds >= 18:
            # クイズ18の結果
            quiz_18_indices = atl.test_indices[min(90, len(atl.test_indices)):min(180, len(atl.test_indices))]
            feedback_18_indices = atl.feedback_indices[min(45, len(atl.feedback_indices)):min(90, len(atl.feedback_indices))] if method == "ATL" else []
            
            if len(atl.integrated_risk_history) > 17 and len(atl.true_risk_history) > 17:
                estimation_error_18 = abs(atl.integrated_risk_history[17] - atl.true_risk_history[17])
            else:
                estimation_error_18 = 0
                
            quiz_18_result = {
                'model': model,
                'labeled_indices': atl.labeled_indices[:min(180, len(atl.labeled_indices))],
                'test_indices': quiz_18_indices,
                'feedback_indices': feedback_18_indices,
                'estimation_error': estimation_error_18
            }
        else:
            quiz_18_result = None
        
        # 結果を保存
        results[method] = {
            'atl': atl,
            'quiz_9': quiz_9_result,
            'quiz_18': quiz_18_result,
            'estimation_errors': [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        }
    
    # クイズ9の結果を並べて可視化
    plt.figure(figsize=(20, 5))
    for i, method in enumerate(methods):
        if results[method]['quiz_9'] is not None:
            plt.subplot(1, 4, i+1)
            plot_quiz_result(
                X, y, 
                results[method]['quiz_9'],
                f"{method} at Quiz 9"
            )
    plt.tight_layout()
    plt.savefig("comparison_quiz_9.png")
    print("Successfully plotted Quiz 9 comparison")
    
    # クイズ18の結果を並べて可視化
    plt.figure(figsize=(20, 5))
    for i, method in enumerate(methods):
        if results[method]['quiz_18'] is not None:
            plt.subplot(1, 4, i+1)
            plot_quiz_result(
                X, y, 
                results[method]['quiz_18'],
                f"{method} at Quiz 18"
            )
    plt.tight_layout()
    plt.savefig("comparison_quiz_18.png")
    print("Successfully plotted Quiz 18 comparison")
    
    # リスク推定誤差を比較
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(
            range(1, len(results[method]['estimation_errors']) + 1),
            results[method]['estimation_errors'],
            marker='o',
            label=method
        )
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_estimation_error.png")
    print("Successfully plotted estimation error comparison")
    
    # 最終的な分類精度を比較
    accuracies = {}
    for method in methods:
        model = results[method]['atl'].model
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        accuracies[method] = acc
        print(f"{method} final accuracy: {acc:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylabel('Accuracy')
    plt.title('Final Classification Accuracy')
    plt.savefig("comparison_accuracy.png")
    print("Successfully plotted accuracy comparison")
    
    return results

def plot_quiz_result(X, y, quiz_result, title):
    """クイズの結果を可視化します。"""
    model = quiz_result['model']
    labeled_indices = quiz_result['labeled_indices']
    test_indices = quiz_result['test_indices']
    feedback_indices = quiz_result['feedback_indices']
    estimation_error = quiz_result['estimation_error']
    
    # 決定境界を可視化
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 格子点での予測確率
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    
    if Z.shape[1] > 1:  # 多クラス分類
        Z = Z[:, 1]
    
    Z = Z.reshape(xx.shape)
    
    # 決定境界をプロット
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
    
    # データ点をプロット
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    
    # ラベル付きサンプルをハイライト
    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], s=100, facecolors='none', edgecolors='g', linewidths=2, label='AL Sample')
    
    # テストサンプルをハイライト
    plt.scatter(X[test_indices, 0], X[test_indices, 1], s=100, facecolors='none', edgecolors='b', linewidths=2, label='AT Sample')
    
    # フィードバックサンプルをハイライト
    if len(feedback_indices) > 0:
        plt.scatter(X[feedback_indices, 0], X[feedback_indices, 1], s=100, facecolors='none', edgecolors='r', linewidths=2, label='Feedback Sample')
    
    plt.title(f"{title}\nEstimation Error:{estimation_error:.1e}")
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)