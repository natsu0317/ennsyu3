import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss
from sklearn.metrics import accuracy_score
import types

def compare_atl_variants(dataset_name, n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3):
    """
    ATL, ATL-NF, ATL-RFの性能を詳細に比較します。
    
    Parameters:
    -----------
    dataset_name : str
        データセット名 ('mnist', 'fashion_mnist', 'cifar10')
    n_rounds : int, optional
        アクティブラーニングのラウンド数
    n_samples_per_round : int, optional
        ラウンドごとに選択するサンプル数
    n_test_per_round : int, optional
        クイズごとのテストサンプル数
    n_runs : int, optional
        実験の実行回数（平均と標準偏差を計算するため）
    
    Returns:
    --------
    results : dict
        比較結果を含む辞書
    """
    print(f"Comparing ATL variants on {dataset_name}...")
    
    # 結果を格納する辞書
    all_results = {
        'ATL': {'accuracy': [], 'risk': [], 'labels': [], 'time': [], 'estimation_error': []},
        'ATL-NF': {'accuracy': [], 'risk': [], 'labels': [], 'time': [], 'estimation_error': []},
        'ATL-RF': {'accuracy': [], 'risk': [], 'labels': [], 'time': [], 'estimation_error': []}
    }
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # データセットを読み込み
        train_dataset, test_dataset = load_dataset(dataset_name)
        
        # アクティブラーニング用にデータを準備
        X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
            train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
        )
        
        methods = ["ATL", "ATL-NF", "ATL-RF"]
        
        for method in methods:
            print(f"Running {method}...")
            
            # モデルを初期化
            if dataset_name == "mnist" or dataset_name == "fashion_mnist":
                input_dim = 784  # 28x28
                hidden_dim = 128
                output_dim = 10
            else:  # cifar10
                input_dim = 3072  # 32x32x3
                hidden_dim = 256
                output_dim = 10
                
            model = NNModel(input_dim, hidden_dim, output_dim, lr=0.001, batch_size=64, epochs=5)
            
            # インデックスを初期化
            initial_indices = list(range(len(X_initial)))
            
            # 開始時間を記録
            start_time = time.time()
            
            # ATLフレームワークを初期化
            atl = ATLFramework(
                model=model,
                X_pool=np.vstack([X_initial, X_pool]),
                y_pool=np.hstack([y_initial, y_pool]),
                initial_labeled_indices=initial_indices,
                test_frequency=1,
                test_batch_size=n_test_per_round,
                feedback_ratio=0.5 if method == "ATL" or method == "ATL-RF" else 0.0,
                window_size=3
            )
            
            # ATL-RFのフィードバック選択を修正
            if method == "ATL-RF":
                # select_feedback_samplesメソッドをランダム選択にオーバーライド
                def random_feedback(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
                    if len(X_test) == 0:
                        return []
                    selected_indices = np.random.choice(len(test_indices), size=n_feedback, replace=False)
                    original_indices = test_indices[selected_indices]
                    self.feedback_indices.extend(original_indices)
                    return original_indices
                
                atl.active_tester.select_feedback_samples = types.MethodType(random_feedback, atl.active_tester)
            
            # アクティブラーニングを実行
            atl.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
            
            # 実行時間を記録
            execution_time = time.time() - start_time
            
            # ホールドアウトセットで評価
            model.fit(atl.X_pool[atl.labeled_indices], atl.y_pool[atl.labeled_indices])
            holdout_probs = model.predict_proba(X_holdout)
            holdout_preds = np.argmax(holdout_probs, axis=1)
            holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
            holdout_accuracy = accuracy_score(y_holdout, holdout_preds)
            
            # ラベル数を計算
            n_labels_used = len(atl.labeled_indices) + len(atl.test_indices)
            
            # 推定誤差を計算（最終ラウンドの値）
            if len(atl.integrated_risk_history) > 0 and len(atl.true_risk_history) > 0:
                final_estimation_error = abs(atl.integrated_risk_history[-1] - atl.true_risk_history[-1])
            else:
                final_estimation_error = float('nan')
            
            # 結果を保存
            all_results[method]['accuracy'].append(holdout_accuracy)
            all_results[method]['risk'].append(holdout_risk)
            all_results[method]['labels'].append(n_labels_used)
            all_results[method]['time'].append(execution_time)
            all_results[method]['estimation_error'].append(final_estimation_error)
    
    # 平均と標準偏差を計算
    summary = {}
    for method in methods:
        summary[method] = {
            'accuracy_mean': np.mean(all_results[method]['accuracy']),
            'accuracy_std': np.std(all_results[method]['accuracy']),
            'risk_mean': np.mean(all_results[method]['risk']),
            'risk_std': np.std(all_results[method]['risk']),
            'labels_mean': np.mean(all_results[method]['labels']),
            'labels_std': np.std(all_results[method]['labels']),
            'time_mean': np.mean(all_results[method]['time']),
            'time_std': np.std(all_results[method]['time']),
            'estimation_error_mean': np.mean(all_results[method]['estimation_error']),
            'estimation_error_std': np.std(all_results[method]['estimation_error'])
        }
    
    # 結果を表示
    print("\n===== Detailed Comparison Results =====")
    
    # テーブル形式で結果を表示
    df = pd.DataFrame({
        'Method': methods,
        'Accuracy': [f"{summary[m]['accuracy_mean']:.4f} ± {summary[m]['accuracy_std']:.4f}" for m in methods],
        'Risk': [f"{summary[m]['risk_mean']:.4f} ± {summary[m]['risk_std']:.4f}" for m in methods],
        'Labels Used': [f"{summary[m]['labels_mean']:.0f} ± {summary[m]['labels_std']:.0f}" for m in methods],
        'Time (s)': [f"{summary[m]['time_mean']:.2f} ± {summary[m]['time_std']:.2f}" for m in methods],
        'Est. Error': [f"{summary[m]['estimation_error_mean']:.4f} ± {summary[m]['estimation_error_std']:.4f}" for m in methods]
    })
    
    print(df.to_string(index=False))
    
    # 相対的な比較
    print("\n===== Relative Comparison =====")
    
    # ATLを基準とした相対的な比較
    relative_df = pd.DataFrame({
        'Method': ["ATL-NF vs ATL", "ATL-RF vs ATL"],
        'Accuracy Diff': [
            f"{summary['ATL-NF']['accuracy_mean'] - summary['ATL']['accuracy_mean']:.4f} ({(summary['ATL-NF']['accuracy_mean'] - summary['ATL']['accuracy_mean']) / summary['ATL']['accuracy_mean'] * 100:.2f}%)",
            f"{summary['ATL-RF']['accuracy_mean'] - summary['ATL']['accuracy_mean']:.4f} ({(summary['ATL-RF']['accuracy_mean'] - summary['ATL']['accuracy_mean']) / summary['ATL']['accuracy_mean'] * 100:.2f}%)"
        ],
        'Risk Diff': [
            f"{summary['ATL-NF']['risk_mean'] - summary['ATL']['risk_mean']:.4f} ({(summary['ATL-NF']['risk_mean'] - summary['ATL']['risk_mean']) / summary['ATL']['risk_mean'] * 100:.2f}%)",
            f"{summary['ATL-RF']['risk_mean'] - summary['ATL']['risk_mean']:.4f} ({(summary['ATL-RF']['risk_mean'] - summary['ATL']['risk_mean']) / summary['ATL']['risk_mean'] * 100:.2f}%)"
        ],
        'Labels Diff': [
            f"{summary['ATL-NF']['labels_mean'] - summary['ATL']['labels_mean']:.0f} ({(summary['ATL-NF']['labels_mean'] - summary['ATL']['labels_mean']) / summary['ATL']['labels_mean'] * 100:.2f}%)",
            f"{summary['ATL-RF']['labels_mean'] - summary['ATL']['labels_mean']:.0f} ({(summary['ATL-RF']['labels_mean'] - summary['ATL']['labels_mean']) / summary['ATL']['labels_mean'] * 100:.2f}%)"
        ],
        'Est. Error Diff': [
            f"{summary['ATL-NF']['estimation_error_mean'] - summary['ATL']['estimation_error_mean']:.4f} ({(summary['ATL-NF']['estimation_error_mean'] - summary['ATL']['estimation_error_mean']) / summary['ATL']['estimation_error_mean'] * 100:.2f}%)",
            f"{summary['ATL-RF']['estimation_error_mean'] - summary['ATL']['estimation_error_mean']:.4f} ({(summary['ATL-RF']['estimation_error_mean'] - summary['ATL']['estimation_error_mean']) / summary['ATL']['estimation_error_mean'] * 100:.2f}%)"
        ]
    })
    
    print(relative_df.to_string(index=False))
    
    # 最良の方法を特定
    best_accuracy = max([summary[m]['accuracy_mean'] for m in methods])
    best_accuracy_method = methods[[summary[m]['accuracy_mean'] for m in methods].index(best_accuracy)]
    
    lowest_risk = min([summary[m]['risk_mean'] for m in methods])
    lowest_risk_method = methods[[summary[m]['risk_mean'] for m in methods].index(lowest_risk)]
    
    lowest_error = min([summary[m]['estimation_error_mean'] for m in methods])
    lowest_error_method = methods[[summary[m]['estimation_error_mean'] for m in methods].index(lowest_error)]
    
    print("\n===== Best Performing Method =====")
    print(f"Best Accuracy: {best_accuracy_method} ({best_accuracy:.4f})")
    print(f"Lowest Risk: {lowest_risk_method} ({lowest_risk:.4f})")
    print(f"Lowest Estimation Error: {lowest_error_method} ({lowest_error:.4f})")
    
    # 結果をグラフで可視化
    plt.figure(figsize=(15, 12))
    
    # 1. 精度比較
    plt.subplot(2, 2, 1)
    accuracy_means = [summary[m]['accuracy_mean'] for m in methods]
    accuracy_stds = [summary[m]['accuracy_std'] for m in methods]
    plt.bar(methods, accuracy_means, yerr=accuracy_stds)
    plt.ylabel('Accuracy')
    plt.title('Holdout Accuracy Comparison')
    plt.ylim(min(accuracy_means) * 0.95, max(accuracy_means) * 1.05)
    
    # 2. リスク比較
    plt.subplot(2, 2, 2)
    risk_means = [summary[m]['risk_mean'] for m in methods]
    risk_stds = [summary[m]['risk_std'] for m in methods]
    plt.bar(methods, risk_means, yerr=risk_stds)
    plt.ylabel('Risk (Cross-Entropy Loss)')
    plt.title('Holdout Risk Comparison')
    
    # 3. ラベル使用量比較
    plt.subplot(2, 2, 3)
    labels_means = [summary[m]['labels_mean'] for m in methods]
    labels_stds = [summary[m]['labels_std'] for m in methods]
    plt.bar(methods, labels_means, yerr=labels_stds)
    plt.ylabel('Number of Labels Used')
    plt.title('Label Usage Comparison')
    
    # 4. 推定誤差比較
    plt.subplot(2, 2, 4)
    error_means = [summary[m]['estimation_error_mean'] for m in methods]
    error_stds = [summary[m]['estimation_error_std'] for m in methods]
    plt.bar(methods, error_means, yerr=error_stds)
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error Comparison')
    
    plt.tight_layout()
    plt.savefig(f"atl_variants_comparison_{dataset_name}.png")
    
    return {
        'all_results': all_results,
        'summary': summary,
        'best_accuracy_method': best_accuracy_method,
        'lowest_risk_method': lowest_risk_method,
        'lowest_error_method': lowest_error_method
    }
