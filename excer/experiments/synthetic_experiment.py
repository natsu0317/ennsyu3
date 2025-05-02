import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.gp_model import GPModel
from datasets.synthetic import create_moon_dataset_with_clusters
from atl_framework.atl import ATLFramework
from utils.visualization import plot_decision_boundary

def run_synthetic_experiment():
    """Run the synthetic data experiment."""
    # Create the dataset
    X, y = create_moon_dataset_with_clusters()
    
    # Split into initial labeled set and pool
    X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, test_size=0.9, random_state=42)
    initial_indices = list(range(len(X_initial)))
    
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
        feedback_ratio=0.5,
        window_size=3
    )
    
    # Run active learning
    atl.run_active_learning(n_rounds=20, n_samples_per_round=10)
    
    # 実際に実行されたラウンド数を取得
    actual_rounds = len(atl.integrated_risk_history)
    print(f"Actual rounds completed: {actual_rounds}")
    
    # 早期停止の場合の処理
    if actual_rounds < 9:
        print("Early stopping occurred before quiz 9. Using the last quiz for visualization.")
        last_quiz_idx = actual_rounds - 1
        
        # Plot decision boundary at the last quiz
        plot_decision_boundary(
            model, X, y, 
            title=f"Decision Boundary at Last Quiz (Round {last_quiz_idx + 1})\nEstimation Error: {abs(atl.integrated_risk_history[last_quiz_idx] - atl.true_risk_history[last_quiz_idx]):.1e}",
            labeled_indices=atl.labeled_indices,
            test_indices=atl.test_indices,
            feedback_indices=atl.feedback_indices
        ).savefig("atl_last_quiz.png")
        
        # Plot risk estimation error
        plt.figure(figsize=(10, 6))
        estimation_errors = [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
        plt.plot(range(1, len(estimation_errors) + 1), estimation_errors, marker='o', label='ATL')
        plt.xlabel('Test Phase')
        plt.ylabel('Estimation Error')
        plt.title('Risk Estimation Error')
        plt.legend()
        plt.grid(True)
        plt.savefig("atl_estimation_error.png")
        
        return atl
    
    # 以下は元のコード（ラウンド数が十分な場合）
    # Plot results at different stages
    quiz_9_indices = atl.test_indices[:min(90, len(atl.test_indices))]
    
    # 十分なラウンド数がある場合のみquiz_18_indicesを設定
    if actual_rounds >= 18:
        quiz_18_indices = atl.test_indices[90:min(180, len(atl.test_indices))]
    else:
        quiz_18_indices = []
    
    feedback_indices = atl.feedback_indices
    
    # Plot decision boundary at quiz 9 (または最大のクイズ)
    max_quiz_idx = min(8, actual_rounds - 1)  # インデックスは0から始まるので-1
    plot_decision_boundary(
        model, X, y, 
        title=f"Decision Boundary at Quiz {max_quiz_idx + 1}\nEstimation Error: {abs(atl.integrated_risk_history[max_quiz_idx] - atl.true_risk_history[max_quiz_idx]):.1e}",
        labeled_indices=atl.labeled_indices[:min(90, len(atl.labeled_indices))],
        test_indices=quiz_9_indices,
        feedback_indices=feedback_indices[:min(45, len(feedback_indices))]
    ).savefig("atl_quiz_9.png")
    
    # Plot decision boundary at quiz 18 (十分なラウンド数がある場合のみ)
    if actual_rounds >= 18:
        plot_decision_boundary(
            model, X, y, 
            title=f"Decision Boundary at Quiz 18\nEstimation Error: {abs(atl.integrated_risk_history[17] - atl.true_risk_history[17]):.1e}",
            labeled_indices=atl.labeled_indices[:min(180, len(atl.labeled_indices))],
            test_indices=quiz_18_indices,
            feedback_indices=feedback_indices[45:min(90, len(feedback_indices))]
        ).savefig("atl_quiz_18.png")
    else:
        # 代わりに最後のクイズを使用
        plot_decision_boundary(
            model, X, y, 
            title=f"Decision Boundary at Last Quiz (Round {actual_rounds})\nEstimation Error: {abs(atl.integrated_risk_history[-1] - atl.true_risk_history[-1]):.1e}",
            labeled_indices=atl.labeled_indices,
            test_indices=atl.test_indices,
            feedback_indices=feedback_indices
        ).savefig("atl_last_quiz.png")
    
    # Plot risk estimation error
    plt.figure(figsize=(10, 6))
    estimation_errors = [abs(est - true) for est, true in zip(atl.integrated_risk_history, atl.true_risk_history)]
    plt.plot(range(1, len(estimation_errors) + 1), estimation_errors, marker='o', label='ATL')
    plt.xlabel('Test Phase')
    plt.ylabel('Estimation Error')
    plt.title('Risk Estimation Error')
    plt.legend()
    plt.grid(True)
    plt.savefig("atl_estimation_error.png")
    
    return atl