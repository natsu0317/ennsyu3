# atl/verification/synthetic_verification.py

import numpy as np
import matplotlib.pyplot as plt
from models.gp_model import GPModel
from datasets.synthetic import create_moon_dataset_with_clusters
from atl_framework.atl import ATLFramework
from utils.visualization import plot_decision_boundary
from sklearn.model_selection import train_test_split
import types

def verify_synthetic_experiment():
    """Verify synthetic experiment results against the paper."""
    # Create the dataset
    X, y = create_moon_dataset_with_clusters()
    
    # Split into initial labeled set and pool
    X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)
    initial_indices = list(range(len(X_initial)))
    
    # Run different methods
    methods = ["AL_sampling", "Random", "ATL-NF", "ATL"]
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
        
        # Modify test selection for different methods
        if method == "AL_sampling":
            # Use AL sampling for test selection
            def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                if excluded_indices is None:
                    excluded_indices = []
                
                # Get predictions for the pool
                probs = self.model.predict_proba(X_pool)
                
                # Calculate entropy for uncertainty
                uncertainties = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                
                # Create a mask for excluded indices
                mask = np.ones(len(X_pool), dtype=bool)
                mask[excluded_indices] = False
                uncertainties[~mask] = -np.inf
                
                # Convert to proposal distribution
                q_star = np.zeros(len(X_pool))
                if np.sum(mask) > 0:
                    # Select top uncertain samples
                    top_indices = np.argsort(uncertainties)[-100:][::-1]
                    q_star[top_indices] = 1
                    q_star = q_star / np.sum(q_star)
                
                return q_star
            
            atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
        
        elif method == "Random":
            # Use random sampling for test selection
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
        
        # Run active learning
        atl.run_active_learning(n_rounds=20, n_samples_per_round=10)
        
        # Store results
        results[method] = {
            'atl': atl,
            'true_risk_history': atl.true_risk_history.copy(),
            'integrated_risk_history': atl.integrated_risk_history.copy(),
            'estimation_errors': [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        }
        
        # Plot decision boundary at quiz 9 and 18
        quiz_9_indices = atl.test_indices[:90]
        quiz_18_indices = atl.test_indices[90:180]
        feedback_indices = atl.feedback_indices
        
        # Plot decision boundary at quiz 9
        error_9 = abs(atl.integrated_risk_history[8] - atl.true_risk_history[8]) if len(atl.integrated_risk_history) > 8 else 0
        plot_decision_boundary(
            model, X, y, 
            title=f"{method} at Quiz 9\nEstimation Error: {error_9:.1e}",
            labeled_indices=atl.labeled_indices[:90],
            test_indices=quiz_9_indices,
            feedback_indices=feedback_indices[:45] if method == "ATL" else None
        ).savefig(f"verification_{method}_quiz_9.png")
        
        # Plot decision boundary at quiz 18
        error_18 = abs(atl.integrated_risk_history[17] - atl.true_risk_history[17]) if len(atl.integrated_risk_history) > 17 else 0
        plot_decision_boundary(
            model, X, y, 
            title=f"{method} at Quiz 18\nEstimation Error: {error_18:.1e}",
            labeled_indices=atl.labeled_indices[:180],
            test_indices=quiz_18_indices,
            feedback_indices=feedback_indices[45:] if method == "ATL" else None
        ).savefig(f"verification_{method}_quiz_18.png")
    
    # Plot estimation error comparison
    plt.figure(figsize=(10, 6))
    for method in methods:
        errors = results[method]['estimation_errors']
        plt.plot(range(1, len(errors) + 1), errors, marker='o', label=method)
    
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("verification_estimation_error_comparison.png")
    
    return results