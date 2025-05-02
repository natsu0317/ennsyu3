import numpy as np
import matplotlib.pyplot as plt
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss


def run_real_data_experiment(dataset_name, n_rounds=20, n_samples_per_round=500, n_test_per_round=100):
    """Run experiment on real-world dataset."""
    print(f"Running experiment on {dataset_name}...")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    # Prepare data for active learning
    X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout = prepare_data_for_al(
        train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000
    )
    
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
        feedback_ratio=0.5,
        window_size=3
    )
    
    # Run active learning
    atl.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
    
    # Evaluate on holdout set
    model.fit(atl.X_pool[atl.labeled_indices], atl.y_pool[atl.labeled_indices])
    holdout_probs = model.predict_proba(X_holdout)
    holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
    
    print(f"Final holdout risk: {holdout_risk:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(atl.true_risk_history) + 1), atl.true_risk_history, marker='o', label='True Risk')
    plt.plot(range(1, len(atl.integrated_risk_history) + 1), atl.integrated_risk_history, marker='x', label='Estimated Risk')
    plt.xlabel('Test Phase')
    plt.ylabel('Risk')
    plt.title(f'{dataset_name} - Risk Estimation')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    estimation_errors = [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
    plt.plot(range(1, len(estimation_errors) + 1), estimation_errors, marker='o')
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"atl_{dataset_name}_results.png")
    
    return atl, holdout_risk