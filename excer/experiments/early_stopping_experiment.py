import numpy as np
import matplotlib.pyplot as plt
from models.nn_model import NNModel
from datasets.image_dataset import load_dataset, prepare_data_for_al
from atl_framework.atl import ATLFramework
from utils.losses import cross_entropy_loss
def run_early_stopping_experiment(dataset_name, n_rounds=40, n_samples_per_round=500, n_test_per_round=100):
    """Run experiment to evaluate early stopping effectiveness."""
    print(f"Running early stopping experiment on {dataset_name}...")
    
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
    
    # Initialize ATL framework with early stopping
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
    
    # Run active learning with early stopping
    atl.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
    
    # Evaluate on holdout set
    model.fit(atl.X_pool[atl.labeled_indices], atl.y_pool[atl.labeled_indices])
    holdout_probs = model.predict_proba(X_holdout)
    holdout_risk = cross_entropy_loss(holdout_probs, y_holdout)
    
    print(f"Final holdout risk with early stopping: {holdout_risk:.4f}")
    print(f"Total rounds used: {len(atl.integrated_risk_history)}")
    
    # Run without early stopping for comparison
    model_no_stop = NNModel(input_dim, hidden_dim, output_dim, lr=0.001, batch_size=64, epochs=5)
    
    # Initialize ATL framework without early stopping
    atl_no_stop = ATLFramework(
        model=model_no_stop,
        X_pool=np.vstack([X_initial, X_pool]),
        y_pool=np.hstack([y_initial, y_pool]),
        initial_labeled_indices=initial_indices,
        test_frequency=1,
        test_batch_size=n_test_per_round,
        feedback_ratio=0.5,
        window_size=3
    )
    
    # Override check_early_stopping to always return False
    atl_no_stop.check_early_stopping = lambda: False
    
    # Run active learning without early stopping
    atl_no_stop.run_active_learning(n_rounds=n_rounds, n_samples_per_round=n_samples_per_round)
    
    # Evaluate on holdout set
    model_no_stop.fit(atl_no_stop.X_pool[atl_no_stop.labeled_indices], atl_no_stop.y_pool[atl_no_stop.labeled_indices])
    holdout_probs_no_stop = model_no_stop.predict_proba(X_holdout)
    holdout_risk_no_stop = cross_entropy_loss(holdout_probs_no_stop, y_holdout)
    
    print(f"Final holdout risk without early stopping: {holdout_risk_no_stop:.4f}")
    print(f"Total rounds used: {len(atl_no_stop.integrated_risk_history)}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(atl.true_risk_history) + 1), atl.true_risk_history, marker='o', label='With Early Stopping')
    plt.plot(range(1, len(atl_no_stop.true_risk_history) + 1), atl_no_stop.true_risk_history, marker='x', label='Without Early Stopping')
    
    # Mark the early stopping point
    stop_point = len(atl.true_risk_history)
    plt.axvline(x=stop_point, color='r', linestyle='--', label=f'Early Stop at Round {stop_point}')
    
    plt.xlabel('Test Phase')
    plt.ylabel('True Risk')
    plt.title(f'{dataset_name} - Early Stopping Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # Calculate the number of labels used
    labels_with_stop = len(atl.labeled_indices) + len(atl.test_indices)
    labels_no_stop = len(atl_no_stop.labeled_indices) + len(atl_no_stop.test_indices)
    
    plt.bar(['With Early Stopping', 'Without Early Stopping'], [labels_with_stop, labels_no_stop])
    plt.ylabel('Number of Labels Used')
    plt.title('Label Efficiency Comparison')
    
    plt.tight_layout()
    plt.savefig(f"atl_{dataset_name}_early_stopping.png")
    
    return atl, atl_no_stop, holdout_risk, holdout_risk_no_stop