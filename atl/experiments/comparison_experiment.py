import numpy as np
import matplotlib.pyplot as plt
import types
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss

"""
ATL-NF: feedback無し
ATL-RF: feedbackに使うsampleをrandomに選択
"""

def run_comparison_experiment(dataset_name, methods=None, n_rounds=20, n_samples_per_round=500, n_test_per_round=100):
    if methods is None:
        methods = ["ATL", "ATL-NF", "ATL-RF"]
    print(f"Running comparison experiment on {dataset_name}...")
    # Load dataset("mnist", "fashion_mnist", "cifar10")
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    # Prepare data for active learning
    X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
        train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
    )
    results = {}
    for method in methods:
        print(f"Running {method}...")
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
            feedback_ratio=0.5 if method == "ATL" or method == "ATL-RF" else 0.0,
            # ATL-NF: feedbackなし
            window_size=3
        )
        
        # Modify feedback selection for ATL-RF
        if method == "ATL-RF":
            # feedback sampleをrandomに選択
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
        
        print(f"{method} final holdout risk: {holdout_risk:.4f}")
        
        # Store results
        results[method] = {
            'atl': atl,
            'holdout_risk': holdout_risk,
            'true_risk_history': atl.true_risk_history.copy(),
            'integrated_risk_history': atl.integrated_risk_history.copy(),
            'estimation_errors': [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        }
    
    # Plot comparison results
    plt.figure(figsize=(12, 12))
    
    # Plot holdout risk
    plt.subplot(3, 1, 1)
    for method in methods:
        plt.plot(range(1, len(results[method]['true_risk_history']) + 1), 
                 results[method]['true_risk_history'], marker='o', label=method)
    plt.xlabel('Test Phase')
    plt.ylabel('True Risk')
    plt.title(f'{dataset_name} - True Risk Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot estimation error
    plt.subplot(3, 1, 2)
    for method in methods:
        plt.plot(range(1, len(results[method]['estimation_errors']) + 1), 
                 results[method]['estimation_errors'], marker='o', label=method)
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot number of labeled samples
    plt.subplot(3, 1, 3)
    for method in methods:
        atl = results[method]['atl']
        n_labeled = [len(atl.labeled_indices) - len(atl.feedback_indices)]
        n_feedback = [len(atl.feedback_indices)]
        
        plt.bar([method], n_labeled, label='AL Samples')
        plt.bar([method], n_feedback, bottom=n_labeled, label='Feedback Samples')
    
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"atl_{dataset_name}_comparison.png")
    
    return results