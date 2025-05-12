import numpy as np
import matplotlib.pyplot as plt
import types
import torch
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss

def run_comparison_experiment(dataset_name, methods=None, n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3):
    """
    ATL, ATL-NF, ATL-RFの性能を比較します。
    論文の表2と表3の結果を再現します。
    
    Parameters:
    -----------
    dataset_name : str
        データセット名 ('mnist', 'fashion_mnist', 'cifar10')
    methods : list, optional
        比較する手法のリスト
    n_rounds : int, optional
        アクティブラーニングのラウンド数
    n_samples_per_round : int, optional
        ラウンドごとに選択するサンプル数
    n_test_per_round : int, optional
        クイズごとのテストサンプル数
    n_runs : int, optional
        実験の実行回数
    
    Returns:
    --------
    results : dict
        比較結果を含む辞書
    """
    if methods is None:
        methods = ["ATL", "ATL-NF", "ATL-RF"]
        
    print(f"Running comparison experiment on {dataset_name}...")
    
    # 結果を格納する辞書
    all_results = {
        method: {
            'holdout_risks': {r: [] for r in [4, 8, 12, 16, 20]},
            'estimation_errors': {r: [] for r in [4, 8, 12, 16, 20]}
        } for method in methods
    }
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # 乱数シードを設定
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)
        
        # データセットを読み込み
        train_dataset, test_dataset = load_dataset(dataset_name)
        
        # アクティブラーニング用にデータを準備
        X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
            train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
        )
        
        for method in methods:
            print(f"  Running {method}...")
            
            # モデルを初期化
            if dataset_name == "mnist" or dataset_name == "fashion_mnist":
                input_dim = 784  # 28x28
                hidden_dim = 128
                output_dim = 10
                epochs = 10  # 論文のC.2.1に基づく
            else:  # cifar10
                input_dim = 3072  # 32x32x3
                hidden_dim = 256
                output_dim = 10
                epochs = 50  # 論文のC.2.1に基づく
                
            model = NNModel(input_dim, hidden_dim, output_dim, dataset_name=dataset_name, lr=0.001, batch_size=64, epochs=epochs)
            
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
                def random_feedback(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
                    if len(X_test) == 0:
                        return []
                    selected_indices = np.random.choice(len(test_indices), size=n_feedback, replace=False)
                    original_indices = test_indices[selected_indices]
                    self.feedback_indices.extend(original_indices)
                    return original_indices
                
                atl.active_tester.select_feedback_samples = types.MethodType(random_feedback, atl.active_tester)
            
            # 特定のラウンドでのリスクと推定誤差を保存
            round_risks = {}
            round_errors = {}
            
            # アクティブラーニングを実行
            for al_round in range(n_rounds):
                print(f"    Active Learning Round {al_round+1}/{n_rounds}")
                
                # 現在のラベル付きデータを取得
                X_labeled, y_labeled = atl.get_labeled_data()
                
                # ラベル付きデータでモデルを学習
                if len(X_labeled) > 0 and y_labeled is not None:
                    model.fit(X_labeled, y_labeled)
                
                # アクティブクイズを実行
                atl.perform_active_quiz(al_round)
                
                # 特定のラウンドでリスクと推定誤差を記録
                if al_round + 1 in [4, 8, 12, 16, 20]:
                    # ホールドアウトセットでリスクを計算
                    holdout_probs = model.predict_proba(X_holdout)
                    holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
                    round_risks[al_round + 1] = holdout_risk
                    
                    # 推定誤差を計算
                    if len(atl.integrated_risk_history) > 0 and len(atl.true_risk_history) > 0:
                        estimation_error = abs(atl.integrated_risk_history[-1] - atl.true_risk_history[-1])
                        round_errors[al_round + 1] = estimation_error
                    
                    print(f"      Round {al_round+1} holdout risk: {holdout_risk:.4f}")
                
                # アクティブラーニングのサンプルを選択
                X_unlabeled, unlabeled_indices = atl.get_unlabeled_data()
                if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
                    print("      No more unlabeled data available.")
                    break
                
                # サンプル数を調整（ATL-NFの場合は追加の50サンプル）
                current_n_samples = n_samples_per_round
                if method == "ATL-NF":
                    current_n_samples = n_samples_per_round + 50  # 論文のC.2.1に基づく
                
                # X_unlabeledに対して選択を行い、実際のインデックスに変換
                selected_indices = atl.active_learner.select_samples(
                    X_unlabeled, n_samples=current_n_samples, 
                    excluded_indices=[]  # X_unlabeledは既に除外済みなので空リスト
                )
                
                # 選択されたインデックスを元のプールのインデックスに変換
                original_indices = [unlabeled_indices[i] for i in selected_indices if i < len(unlabeled_indices)]
                
                # 選択されたサンプルがない場合の処理
                if not original_indices:
                    print("      No more informative samples available.")
                    break
                
                # ラベル付きインデックスを更新
                atl.labeled_indices.extend(original_indices)
                atl.active_learner.labeled_indices.extend(original_indices)
            
            # 結果を保存
            for r in [4, 8, 12, 16, 20]:
                if r in round_risks:
                    all_results[method]['holdout_risks'][r].append(round_risks[r])
                if r in round_errors:
                    all_results[method]['estimation_errors'][r].append(round_errors[r])
    
    # 表2の再現：ホールドアウトテストリスク
    print("\nTable 2: Hold-out test risk using different feedback criteria over 20 AL rounds")
    print("Dataset Method    AL round 4    8           12          16          20")
    print(f"{dataset_name}")
    
    for method in methods:
        means = []
        stds = []
        for r in [4, 8, 12, 16, 20]:
            if all_results[method]['holdout_risks'][r]:
                mean = np.mean(all_results[method]['holdout_risks'][r])
                std = np.std(all_results[method]['holdout_risks'][r])
                means.append(f"{mean:.2f} ± {std:.2f}")
            else:
                means.append("N/A")
        print(f"{method:8} {means[0]:12} {means[1]:12} {means[2]:12} {means[3]:12} {means[4]:12}")
    
    # 表3の再現：フィードバック付きの推定誤差
    print("\nTable 3: Estimation error with feedback over 20 AL rounds (×10^-3)")
    print("Dataset Method    AL round 4    8           12          16          20")
    print(f"{dataset_name}")
    
    for method in methods:
        if method in ["ATL-RF", "ATL"]:  # 表3はATL-RFとATLのみ
            means = []
            stds = []
            for r in [4, 8, 12, 16, 20]:
                if all_results[method]['estimation_errors'][r]:
                    # ×10^-3に変換
                    values = [e * 1000 for e in all_results[method]['estimation_errors'][r]]
                    mean = np.mean(values)
                    std = np.std(values)
                    means.append(f"{mean:.2f} ± {std:.2f}")
                else:
                    means.append("N/A")
            print(f"{method:8} {means[0]:12} {means[1]:12} {means[2]:12} {means[3]:12} {means[4]:12}")
    
    # 結果をグラフで可視化
    plt.figure(figsize=(15, 10))
    
    # ホールドアウトリスクのプロット
    plt.subplot(2, 1, 1)
    x = [4, 8, 12, 16, 20]
    for method in methods:
        means = []
        stds = []
        for r in x:
            if all_results[method]['holdout_risks'][r]:
                means.append(np.mean(all_results[method]['holdout_risks'][r]))
                stds.append(np.std(all_results[method]['holdout_risks'][r]))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        plt.errorbar(x, means, yerr=stds, marker='o', label=method)
    
    plt.xlabel('AL Round')
    plt.ylabel('Hold-out Test Risk')
    plt.title(f'{dataset_name} - Hold-out Test Risk')
    plt.legend()
    plt.grid(True)
    
    # 推定誤差のプロット
    plt.subplot(2, 1, 2)
    for method in methods:
        means = []
        stds = []
        for r in x:
            if all_results[method]['estimation_errors'][r]:
                # ×10^-3に変換
                values = [e * 1000 for e in all_results[method]['estimation_errors'][r]]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        plt.errorbar(x, means, yerr=stds, marker='o', label=method)
    
    plt.xlabel('AL Round')
    plt.ylabel('Estimation Error (×10^-3)')
    plt.title('Risk Estimation Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"atl_{dataset_name}_comparison.png")
    
    return all_results