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
from experiments.comparison_experiment import run_comparison_experiment
from experiments.compare_with_full_training import compare_with_full_training
def main():
    """Run all experiments."""
    # Run synthetic experiment
    print("Running synthetic experiment...")
    atl_synthetic = run_synthetic_experiment()
    
    # Run real-data experiments
    datasets = ["mnist", "fashion_mnist", "cifar10"]
    
    for dataset in datasets:
        
        # Run comparison experiment
        results = run_comparison_experiment(
            dataset, methods=["ATL", "ATL-NF", "ATL-RF"], 
            n_rounds=20, n_samples_per_round=500, n_test_per_round=100
        )

        results = compare_with_full_training(
            dataset, methods=["ATL", "ATL-NF", "ATL-RF"], 
            n_rounds=20, n_samples_per_round=500, n_test_per_round=100
        )

if __name__ == "__main__":
    main()