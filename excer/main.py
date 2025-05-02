import numpy as np
import torch # type: ignore
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

from experiments.synthetic_experiment import run_synthetic_experiment
from experiments.real_data_experiment import run_real_data_experiment
from experiments.comparison_experiment import run_comparison_experiment
from experiments.early_stopping_experiment import run_early_stopping_experiment
from experiments.ablation_study import run_feedback_size_ablation
def main():
    """Run all experiments."""
    # Run synthetic experiment
    print("Running synthetic experiment...")
    atl_synthetic = run_synthetic_experiment()
    
    # Run real-data experiments
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    
    for dataset in datasets:
        # Run single experiment
        atl, holdout_risk = run_real_data_experiment(
            dataset, n_rounds=20, n_samples_per_round=500, n_test_per_round=100
        )
        
        # Run comparison experiment
        results = run_comparison_experiment(
            dataset, methods=["ATL", "ATL-NF", "ATL-RF"], 
            n_rounds=20, n_samples_per_round=500, n_test_per_round=100
        )
        
        # Run early stopping experiment
        atl_es, atl_no_es, risk_es, risk_no_es = run_early_stopping_experiment(
            dataset, n_rounds=40, n_samples_per_round=500, n_test_per_round=100
        )
        
        # Run feedback size ablation
        feedback_results = run_feedback_size_ablation(
            dataset, feedback_sizes=[0.2, 0.25, 0.5, 0.67, 0.83], 
            n_rounds=20, n_samples_per_round=500, n_test_per_round=100
        )

if __name__ == "__main__":
    main()