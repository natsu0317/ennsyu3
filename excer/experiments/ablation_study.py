import numpy as np
import matplotlib.pyplot as plt
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss

def run_feedback_size_ablation(dataset_name, feedback_sizes=None, n_rounds=20, n_samples_per_round=500, n_test_per_round=100):
    """Run ablation study on different feedback sizes."""
    if feedback_sizes is None:
        feedback_sizes = [0.2, 0.25, 0.5, 0.67, 0.83]
        
    print(f"Running feedback size ablation on {dataset_name}...")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    # Prepare data for active learning
    X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
        train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
    )
    
    results = {}
    
    for feedback_ratio in feedback_sizes:
        print(f"Running with feedback ratio {feedback_ratio}...")
        
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
            feedback_ratio=feedback_ratio,
            window_size=3
        )
        
        # Run active learning
        atl.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
        
        # Evaluate on holdout set
        model.fit(atl.X_pool[atl.labeled_indices], atl.y_pool[atl.labeled_indices])
        holdout_probs = model.predict_proba(X_holdout)
        holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
        
        print(f"Feedback ratio {feedback_ratio} final holdout risk: {holdout_risk:.4f}")
        
        # Store results
        results[feedback_ratio] = {
            'atl': atl,
            'holdout_risk': holdout_risk,
            'true_risk_history': atl.true_risk_history.copy(),
            'integrated_risk_history': atl.integrated_risk_history.copy(),
            'estimation_errors': [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        }
    
    # Plot comparison results
    plt.figure(figsize=(15, 12))
    
    # Plot holdout risk
    plt.subplot(3, 1, 1)
    for ratio in feedback_sizes:
        plt.plot(range(1, len(results[ratio]['true_risk_history']) + 1), 
                 results[ratio]['true_risk_history'], marker='o', label=f'Ratio {ratio}')
    plt.xlabel('Test Phase')
    plt.ylabel('True Risk')
    plt.title(f'{dataset_name} - True Risk vs Feedback Ratio')
    plt.legend()
    plt.grid(True)
    
    # Plot estimation error
    plt.subplot(3, 1, 2)
    for ratio in feedback_sizes:
        plt.plot(range(1, len(results[ratio]['estimation_errors']) + 1), 
                 results[ratio]['estimation_errors'], marker='o', label=f'Ratio {ratio}')
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error vs Feedback Ratio')
    plt.legend()
    plt.grid(True)
    
    # Plot final risks as bar chart
    plt.subplot(3, 1, 3)
    ratios = list(results.keys())
    final_risks = [results[ratio]['holdout_risk'] for ratio in ratios]
    final_errors = [results[ratio]['estimation_errors'][-1] for ratio in ratios]
    
    x = np.arange(len(ratios))
    width = 0.35
    
    plt.bar(x - width/2, final_risks, width, label='Final Risk')
    plt.bar(x + width/2, final_errors, width, label='Final Estimation Error')
    
    plt.xlabel('Feedback Ratio')
    plt.ylabel('Value')
    plt.title('Final Risk and Estimation Error vs Feedback Ratio')
    plt.xticks(x, [str(r) for r in ratios])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"atl_{dataset_name}_feedback_ablation.png")
    
    return results