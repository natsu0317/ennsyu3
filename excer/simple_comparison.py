import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.gp_model import GPModel
from datasets.synthetic import create_moon_dataset_with_clusters
from atl_framework.atl import ATLFramework
import types

def run_simple_comparison():
    """Run a simple comparison between ATL and random sampling."""
    print("Running simple comparison between ATL and random sampling...")
    
    # Create the dataset
    X, y = create_moon_dataset_with_clusters()
    
    # Split into initial labeled set and pool
    X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)
    initial_indices = list(range(len(X_initial)))
    
    # Results storage
    methods = ["ATL", "Random"]
    results = {}
    
    for method in methods:
        print(f"Running {method}...")
        
        # Initialize the model
        model = GPModel()
        
        # Initialize ATL framework
        atl = ATLFramework(
            model=model,
            X_pool=X,
            y_pool=y,
            initial_labeled_indices=initial_indices,
            test_frequency=1,
            test_batch_size=10,
            feedback_ratio=0.5 if method == "ATL" else 0.0,
            window_size=3
        )
        
        # Modify test selection for Random method
        if method == "Random":
            # Override compute_test_proposal to use random sampling
            def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                if excluded_indices is None:
                    excluded_indices = []
                
                # Create a mask for excluded indices
                mask = np.ones(len(X_pool), dtype=bool)
                mask[excluded_indices] = False
                
                # Uniform distribution over non-excluded indices
                q_star = np.zeros(len(X_pool))
                q_star[mask] = 1
                if np.sum(mask) > 0:
                    q_star = q_star / np.sum(q_star)
                
                return q_star
            
            atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
        
        # Run active learning for a small number of rounds
        atl.run_active_learning(n_rounds=10, n_samples_per_round=10)
        
        # Store results
        results[method] = {
            'true_risk_history': atl.true_risk_history.copy(),
            'integrated_risk_history': atl.integrated_risk_history.copy(),
            'estimation_errors': [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        }
    
    # Plot estimation error comparison
    plt.figure(figsize=(10, 6))
    for method in methods:
        errors = results[method]['estimation_errors']
        plt.plot(range(1, len(errors) + 1), errors, marker='o', label=method)
    
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error Comparison: ATL vs Random')
    plt.legend()
    plt.grid(True)
    plt.savefig("simple_comparison_error.png")
    
    # Print summary statistics
    print("\nSummary of estimation errors:")
    for method in methods:
        errors = results[method]['estimation_errors']
        print(f"{method}:")
        print(f"  Mean error: {np.mean(errors):.4f}")
        print(f"  Final error: {errors[-1]:.4f}")
        print(f"  Max error: {np.max(errors):.4f}")
    
    return results

if __name__ == "__main__":
    results = run_simple_comparison()