# atl/verification/real_data_verification.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..models.nn_model import NNModel
from ..datasets.image_dataset import load_dataset, prepare_data_for_al
from ..atl_framework.atl import ATLFramework
from ..utils.losses import cross_entropy_loss
import types

def verify_real_data_experiment(dataset_name, n_rounds=20, n_samples_per_round=500, n_test_per_round=100, n_runs=3):
    """Verify real data experiment results against the paper."""
    print(f"Verifying {dataset_name} results...")
    
    # Methods to compare
    methods = ["ARE", "AT", "ASE", "ATL-NF", "ATL-RF", "ATL"]
    
    # Results storage
    estimation_errors = {method: [] for method in methods}
    holdout_risks = {method: [] for method in methods}
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # Load dataset
        train_dataset, test_dataset = load_dataset(dataset_name)
        
        # Prepare data for active learning
        X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
            train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
        )
        
        for method in methods:
            print(f"  Running {method}...")
            
            # Initialize model
            if dataset_name == "mnist" or dataset_name == "fashion_mnist":
                input_dim = 784  # 28x28
                hidden_dim = 128
                output_dim = 10
            else:  # cifar10
                input_dim = 3072  # 32x32x3
                hidden_dim = 256
                output_dim = 10
                
            model = NNModel(input_dim, hidden_dim, output_dim, lr=0.001, batch_size=64, epochs=5)
            
            # Initialize indices
            initial_indices = list(range(len(X_initial)))
            
            # Initialize ATL framework
            atl = ATLFramework(
                model=model,
                X_pool=np.vstack([X_initial, X_pool]),
                y_pool=np.hstack([y_initial, y_pool]),
                initial_labeled_indices=initial_indices,
                test_frequency=1,
                test_batch_size=n_test_per_round,
                feedback_ratio=0.5 if method in ["ATL", "ATL-RF"] else 0.0,
                window_size=3
            )
            
            # Modify test selection for different methods
            if method == "ARE":
                # Use ARE quiz without integration
                def integrated_risk_estimation(self, X_quizzes, y_quizzes, quiz_weights, model=None):
                    # Just use the latest quiz
                    if not X_quizzes or not y_quizzes:
                        return 0
                    return self.estimate_risk(X_quizzes[-1], y_quizzes[-1], quiz_weights[-1])
                
                atl.active_tester.integrated_risk_estimation = types.MethodType(integrated_risk_estimation, atl.active_tester)
            
            elif method == "AT":
                # Implement AT integrate method
                def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                    if excluded_indices is None:
                        excluded_indices = []
                    
                    # Create a surrogate model to predict loss
                    # For simplicity, we'll use the current model's uncertainty as a proxy
                    probs = self.model.predict_proba(X_pool)
                    uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    
                    # Create a mask for excluded indices
                    mask = np.ones(len(X_pool), dtype=bool)
                    mask[excluded_indices] = False
                    uncertainty[~mask] = -np.inf
                    
                    # Convert to proposal distribution
                    q_star = np.zeros(len(X_pool))
                    if np.sum(mask) > 0:
                        # Select top uncertain samples
                        top_indices = np.argsort(uncertainty)[-100:][::-1]
                        q_star[top_indices] = 1
                        q_star = q_star / np.sum(q_star)
                    
                    return q_star
                
                atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
            
            elif method == "ASE":
                # Implement ASE integrate method
                def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
                    if excluded_indices is None:
                        excluded_indices = []
                    
                    # Create a surrogate model to predict loss
                    # For simplicity, we'll use the current model's uncertainty as a proxy
                    probs = self.model.predict_proba(X_pool)
                    uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    
                    # Create a mask for excluded indices
                    mask = np.ones(len(X_pool), dtype=bool)
                    mask[excluded_indices] = False
                    uncertainty[~mask] = -np.inf
                    
                    # Convert to proposal distribution based on predicted loss
                    q_star = uncertainty.copy()
                    q_star[~mask] = 0
                    if np.sum(q_star) > 0:
                        q_star = q_star / np.sum(q_star)
                    
                    return q_star
                
                atl.active_tester.compute_test_proposal = types.MethodType(compute_test_proposal, atl.active_tester)
                
                def integrated_risk_estimation(self, X_quizzes, y_quizzes, quiz_weights, model=None):
                    # Use surrogate model for risk estimation
                    # For simplicity, we'll use the average of all quiz results
                    if not X_quizzes or not y_quizzes:
                        return 0
                    
                    all_X = np.vstack(X_quizzes)
                    all_y = np.hstack(y_quizzes)
                    all_weights = np.hstack(quiz_weights)
                    
                    return self.estimate_risk(all_X, all_y, all_weights)
                
                atl.active_tester.integrated_risk_estimation = types.MethodType(integrated_risk_estimation, atl.active_tester)
            
            elif method == "ATL-RF":
                # Override the select_feedback_samples method to use random selection
                def random_feedback(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
                    if len(X_test) == 0:
                        return []
                    selected_indices = np.random.choice(len(test_indices), size=n_feedback, replace=False)
                    original_indices = test_indices[selected_indices]
                    self.feedback_indices.extend(original_indices)
                    return original_indices
                
                atl.active_tester.select_feedback_samples = types.MethodType(random_feedback, atl.active_tester)
            
            # Run active learning
            atl.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
            
            # Evaluate on holdout set
            model.fit(atl.X_pool[atl.labeled_indices], atl.y_pool[atl.labeled_indices])
            holdout_probs = model.predict_proba(X_holdout)
            holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
            
            # Store results
            errors = [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
            estimation_errors[method].append(errors)
            holdout_risks[method].append(holdout_risk)
    
    # Calculate mean and standard deviation of estimation errors
    mean_errors = {}
    std_errors = {}
    
    for method in methods:
        # Ensure all runs have the same length
        min_len = min(len(errors) for errors in estimation_errors[method])
        aligned_errors = [errors[:min_len] for errors in estimation_errors[method]]
        
        # Calculate mean and std
        mean_errors[method] = np.mean(aligned_errors, axis=0)
        std_errors[method] = np.std(aligned_errors, axis=0)
    
    # Create tables similar to the paper
    rounds_to_check = [4, 8, 12, 16, 20]
    
    # Table 1: Estimation error
    table1_data = []
    for method in ["ARE", "AT", "ASE", "ATL-NF"]:
        row = [method]
        for round_idx in rounds_to_check:
            if round_idx - 1 < len(mean_errors[method]):
                value = mean_errors[method][round_idx - 1] * 1000  # Convert to x10^-3
                std = std_errors[method][round_idx - 1] * 1000
                row.append(f"{value:.2f} ± {std:.2f}")
            else:
                row.append("N/A")
        table1_data.append(row)
    
    table1 = pd.DataFrame(table1_data, columns=["Method"] + [f"Round {r}" for r in rounds_to_check])
    print("\nTable 1: Estimation Error (x10^-3)")
    print(table1.to_string(index=False))
    
    # Table 2: Hold-out test risk
    table2_data = []
    for method in ["ATL-NF", "ATL-RF", "ATL"]:
        row = [method]
        for round_idx in rounds_to_check:
            if round_idx - 1 < len(mean_errors[method]):
                # Use true risk as a proxy for holdout risk
                value = np.mean([run.true_risk_history[round_idx - 1] for run in [results[method]['atl'] for results in [{'atl': atl}]]])
                std = np.std([run.true_risk_history[round_idx - 1] for run in [results[method]['atl'] for results in [{'atl': atl}]]])
                row.append(f"{value:.2f} ± {std:.2f}")
            else:
                row.append("N/A")
        table2_data.append(row)
    
    table2 = pd.DataFrame(table2_data, columns=["Method"] + [f"Round {r}" for r in rounds_to_check])
    print("\nTable 2: Hold-out Test Risk")
    print(table2.to_string(index=False))
    
    # Table 3: Estimation error with feedback
    table3_data = []
    for method in ["ATL-RF", "ATL"]:
        row = [method]
        for round_idx in rounds_to_check:
            if round_idx - 1 < len(mean_errors[method]):
                value = mean_errors[method][round_idx - 1] * 1000  # Convert to x10^-3
                std = std_errors[method][round_idx - 1] * 1000
                row.append(f"{value:.2f} ± {std:.2f}")
            else:
                row.append("N/A")
        table3_data.append(row)
    
    table3 = pd.DataFrame(table3_data, columns=["Method"] + [f"Round {r}" for r in rounds_to_check])
    print("\nTable 3: Estimation Error with Feedback (x10^-3)")
    print(table3.to_string(index=False))
    
    # Plot estimation error comparison
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.plot(range(1, len(mean_errors[method]) + 1), mean_errors[method], marker='o', label=method)
        plt.fill_between(
            range(1, len(mean_errors[method]) + 1),
            mean_errors[method] - std_errors[method],
            mean_errors[method] + std_errors[method],
            alpha=0.2
        )
    
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title(f'{dataset_name} - Risk Estimation Error Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"verification_{dataset_name}_estimation_error.png")
    
    return {
        'estimation_errors': estimation_errors,
        'holdout_risks': holdout_risks,
        'mean_errors': mean_errors,
        'std_errors': std_errors
    }