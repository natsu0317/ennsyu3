import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import random
import warnings
warnings.filterwarnings('ignore')

# 乱数シードを固定
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

from experiments.synthetic_experiment import run_synthetic_comparison_experiment
from experiments.real_data_experiment import run_real_data_experiment
from experiments.comparison_experiment import run_comparison_experiment
# from experiments.early_stopping_experiment import run_early_stopping_experiment

def main():
    """すべての実験を実行します。"""
    # 合成データ実験を実行
    print("Running synthetic experiment...")
    synthetic_results = run_synthetic_comparison_experiment()
    
    # 実データ実験を実行
    # datasets = ["mnist", "fashion_mnist", "cifar10"]
    datasets = ["fashion_mnist"]
    
    # for dataset in datasets:
        # 比較実験を実行
        # print(f"\nRunning comparison experiment on {dataset}...")
        # comparison_results = run_comparison_experiment(
        #     dataset, methods=["ATL", "ATL-NF", "ATL-RF"], 
        #     n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3
        # )
        
        # 早期停止実験を実行
        # print(f"\nRunning early stopping experiment on {dataset}...")
        # early_stopping_results = run_early_stopping_experiment(
        #     dataset, n_rounds=40, n_samples_per_round=500, n_test_per_round=100, n_runs=3
        # )

if __name__ == "__main__":
    main()