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

def compare_atl_variants(dataset_name, rounds_to_check=[4, 8, 12, 16, 20], n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3):
    """
    ATL, ATL-NF, ATL-RFの性能を詳細に比較します。
    
    Parameters:
    -----------
    dataset_name : str
        データセット名 ('mnist', 'fashion_mnist', 'cifar10')
    rounds_to_check : list, optional
        リスクを記録するラウンド
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
        'ATL': {'risks_by_round': {r: [] for r in rounds_to_check}},
        'ATL-NF': {'risks_by_round': {r: [] for r in rounds_to_check}},
        'ATL-RF': {'risks_by_round': {r: [] for r in rounds_to_check}}
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
            
            # 早期停止を無効化
            def no_early_stopping(self):
                return False
            
            atl.check_early_stopping = types.MethodType(no_early_stopping, atl)
            
            # 各ラウンドでのリスクを保存するための辞書
            round_risks = {}
            
            # アクティブラーニングを実行
            for al_round in range(n_rounds):
                print(f"  Active Learning Round {al_round+1}/{n_rounds}")
                
                # 現在のラベル付きデータを取得
                X_labeled, y_labeled = atl.get_labeled_data()
                
                # ラベル付きデータでモデルを学習
                if len(X_labeled) > 0 and y_labeled is not None:
                    model.fit(X_labeled, y_labeled)
                
                # テスト頻度に応じてアクティブテストを実行
                if al_round % atl.test_frequency == 0:
                    print(f"    Performing active quiz...")
                    atl.perform_active_quiz(al_round)
                
                # 特定のラウンドでホールドアウトリスクを計算
                if al_round + 1 in rounds_to_check:
                    # ホールドアウトセットで評価
                    holdout_probs = model.predict_proba(X_holdout)
                    holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
                    round_risks[al_round + 1] = holdout_risk
                    print(f"    Round {al_round+1} holdout risk: {holdout_risk:.4f}")
                
                # アクティブラーニングのサンプルを選択
                X_unlabeled, unlabeled_indices = atl.get_unlabeled_data()
                if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
                    print("    No more unlabeled data available.")
                    break
                
                # X_unlabeledに対して選択を行い、実際のインデックスに変換
                selected_indices = atl.active_learner.select_samples(
                    X_unlabeled, n_samples=n_samples_per_round, 
                    excluded_indices=[]  # X_unlabeledは既に除外済みなので空リスト
                )
                
                # 選択されたインデックスを元のプールのインデックスに変換
                original_indices = [unlabeled_indices[i] for i in selected_indices if i < len(unlabeled_indices)]
                
                # 選択されたサンプルがない場合の処理
                if not original_indices:
                    print("    No more informative samples available.")
                    break
                
                # ラベル付きインデックスを更新
                atl.labeled_indices.extend(original_indices)
                atl.active_learner.labeled_indices.extend(original_indices)
                
                # 現在の統計を表示
                if atl.y_pool is not None and len(atl.integrated_risk_history) > 0:
                    print(f"    Labeled samples: {len(atl.labeled_indices)}")
                    print(f"    Test samples: {len(atl.test_indices)}")
                    print(f"    Feedback samples: {len(atl.feedback_indices)}")
            
            # 各ラウンドでのリスクを保存
            for r in rounds_to_check:
                if r in round_risks:
                    all_results[method]['risks_by_round'][r].append(round_risks[r])
                else:
                    # ラウンドが実行されなかった場合はNaNを保存
                    all_results[method]['risks_by_round'][r].append(float('nan'))
    
    # 平均と標準偏差を計算
    summary = {}
    for method in methods:
        summary[method] = {
            'risks_by_round': {}
        }
        for r in rounds_to_check:
            risks = all_results[method]['risks_by_round'][r]
            risks = [r for r in risks if not np.isnan(r)]  # NaNを除外
            if risks:
                summary[method]['risks_by_round'][r] = {
                    'mean': np.mean(risks),
                    'std': np.std(risks)
                }
            else:
                summary[method]['risks_by_round'][r] = {
                    'mean': float('nan'),
                    'std': float('nan')
                }
    
    # 結果を表示
    print("\n===== Hold-out Test Risk by Round =====")
    
    # テーブル形式で結果を表示（論文の表2と同様のフォーマット）
    data = []
    for method in methods:
        row = [method]
        for r in rounds_to_check:
            if not np.isnan(summary[method]['risks_by_round'][r]['mean']):
                row.append(f"{summary[method]['risks_by_round'][r]['mean']:.2f} ± {summary[method]['risks_by_round'][r]['std']:.2f}")
            else:
                row.append("N/A")
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Method'] + [f"Round {r}" for r in rounds_to_check])
    print(df.to_string(index=False))
    
    # 結果をCSVに保存
    df.to_csv(f"atl_holdout_risk_{dataset_name}.csv", index=False)
    
    # グラフで可視化
    plt.figure(figsize=(12, 6))
    
    for method in methods:
        means = []
        stds = []
        for r in rounds_to_check:
            if not np.isnan(summary[method]['risks_by_round'][r]['mean']):
                means.append(summary[method]['risks_by_round'][r]['mean'])
                stds.append(summary[method]['risks_by_round'][r]['std'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        plt.errorbar(rounds_to_check, means, yerr=stds, marker='o', label=method)
    
    plt.xlabel('AL Round')
    plt.ylabel('Hold-out Test Risk')
    plt.title(f'Hold-out Test Risk by Round ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"atl_holdout_risk_{dataset_name}.png")
    
    return {
        'all_results': all_results,
        'summary': summary
    }