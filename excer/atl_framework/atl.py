import numpy as np
import copy
from active_learning.uncertainty_learner import UncertaintyActiveLearner
from active_testing.active_tester import ActiveTester
from utils.losses import cross_entropy_loss

class ATLFramework:
    """Active Testing while Learning Framework."""
    def __init__(self, model, X_pool, y_pool=None, initial_labeled_indices=None, 
                 test_frequency=1, test_batch_size=10, feedback_ratio=0.5, window_size=3):
        self.model = model
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.active_learner = UncertaintyActiveLearner(model)
        self.active_tester = ActiveTester(model)
        
        # Initialize labeled indices
        if initial_labeled_indices is None:
            self.labeled_indices = []
        else:
            self.labeled_indices = list(initial_labeled_indices)
            self.active_learner.labeled_indices = list(initial_labeled_indices)
        
        # Initialize test and feedback indices
        self.test_indices = []
        self.feedback_indices = []
        
        # Parameters
        self.test_frequency = test_frequency  # How often to perform testing
        self.test_batch_size = test_batch_size  # Number of test samples per quiz
        self.feedback_ratio = feedback_ratio  # Ratio of test samples to use as feedback
        self.window_size = window_size  # Window size for moving average in early stopping
        
        # History
        self.risk_history = []
        self.integrated_risk_history = []
        self.true_risk_history = []  # Only for evaluation purposes
        self.quiz_X = []
        self.quiz_y = []
        self.quiz_weights = []
        
    def get_labeled_data(self):
        """Get the current labeled data."""
        X_labeled = self.X_pool[self.labeled_indices]
        y_labeled = self.y_pool[self.labeled_indices] if self.y_pool is not None else None
        return X_labeled, y_labeled
    
    def get_test_data(self):
        """Get the current test data."""
        X_test = self.X_pool[self.test_indices]
        y_test = self.y_pool[self.test_indices] if self.y_pool is not None else None
        return X_test, y_test
    
    def get_feedback_data(self):
        """Get the current feedback data."""
        X_feedback = self.X_pool[self.feedback_indices]
        y_feedback = self.y_pool[self.feedback_indices] if self.y_pool is not None else None
        return X_feedback, y_feedback
    
    def get_unlabeled_indices(self):
        """Get indices of unlabeled data."""
        all_labeled = set(self.labeled_indices + self.test_indices)
        return [i for i in range(len(self.X_pool)) if i not in all_labeled]
    
    def get_unlabeled_data(self):
        """Get the current unlabeled data."""
        unlabeled_indices = self.get_unlabeled_indices()
        X_unlabeled = self.X_pool[unlabeled_indices]
        return X_unlabeled, unlabeled_indices
    
    def perform_active_quiz(self, al_round):
        """Perform an active quiz to evaluate the model."""
        # Get unlabeled data
        X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
        
        if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
            return []
        
        # Get current labeled and test data
        X_labeled, y_labeled = self.get_labeled_data()
        X_test, y_test = self.get_test_data()
        
        # Compute multi-source risk estimate
        multi_source_risk = self.active_tester.compute_multi_source_risk(
            X_labeled, y_labeled, X_test, y_test, X_unlabeled
        )
        
        # Compute test proposal
        excluded_indices = self.labeled_indices + self.test_indices
        q_proposal = self.active_tester.compute_test_proposal(
            self.X_pool, excluded_indices, multi_source_risk
        )
        
        # Select test samples
        selected_indices, weights = self.active_tester.select_test_samples(
            self.X_pool, q_proposal, n_samples=self.test_batch_size, excluded_indices=excluded_indices
        )
        
        # Update test indices
        self.test_indices.extend(selected_indices)
        
        # Get the selected test data
        X_quiz = self.X_pool[selected_indices]
        y_quiz = self.y_pool[selected_indices] if self.y_pool is not None else None
        
        # Store quiz data for later integration
        self.quiz_X.append(X_quiz)
        self.quiz_y.append(y_quiz)
        self.quiz_weights.append(weights)
        
        # Estimate risk for this quiz
        quiz_risk = self.active_tester.estimate_risk(X_quiz, y_quiz, weights)
        self.risk_history.append(quiz_risk)
        
        # Compute integrated risk estimate
        integrated_risk = self.active_tester.integrated_risk_estimation(
            self.quiz_X, self.quiz_y, self.quiz_weights
        )
        self.integrated_risk_history.append(integrated_risk)
        
        # If true labels are available, compute true risk for evaluation
        if self.y_pool is not None:
            true_risk = cross_entropy_loss(self.model.predict_proba(self.X_pool), self.y_pool)
            self.true_risk_history.append(true_risk)
        
        # Select feedback samples
        n_feedback = int(self.feedback_ratio * len(selected_indices))
        if n_feedback > 0:
            feedback_indices = self.active_tester.select_feedback_samples(
                X_quiz, y_quiz, X_labeled, selected_indices, q_proposal, n_feedback=n_feedback
            )
            
            # Update feedback indices
            self.feedback_indices.extend(feedback_indices)
            
            # Add feedback samples to labeled set
            self.labeled_indices.extend(feedback_indices)
            self.active_learner.labeled_indices.extend(feedback_indices)
            
            # Remove feedback samples from test indices
            self.test_indices = [idx for idx in self.test_indices if idx not in feedback_indices]
        
        return selected_indices
    
    def check_early_stopping(self):
        """Check if early stopping criteria are met."""
        if len(self.integrated_risk_history) < self.window_size + 1:
            return False
            
        # Compute change in moving average of integrated risk
        window = self.window_size
        current_avg = np.mean(self.integrated_risk_history[-window:])
        previous_avg = np.mean(self.integrated_risk_history[-(window+1):-1])
        delta_risk = abs(current_avg - previous_avg)
        
        # Get unlabeled data
        X_unlabeled, _ = self.get_unlabeled_data()
        
        # Compute stabilized predictions (SP)
        if len(X_unlabeled) > 0:
            # Get current and previous model predictions
            current_probs = self.model.predict_proba(X_unlabeled)
            
            # Train a temporary model without the most recent samples
            temp_model = copy.deepcopy(self.model)
            X_prev_labeled = self.X_pool[self.labeled_indices[:-self.test_batch_size]]
            y_prev_labeled = self.y_pool[self.labeled_indices[:-self.test_batch_size]]
            temp_model.fit(X_prev_labeled, y_prev_labeled)
            
            # Get previous predictions
            prev_probs = temp_model.predict_proba(X_unlabeled)
            
            # Calculate prediction changes
            pred_changes = np.mean(np.abs(current_probs - prev_probs))
            sp = 1 - pred_changes
        else:
            sp = 1.0
            
        # Combined stopping criterion
        threshold = 0.01  # Threshold for risk change
        sp_threshold = 0.95  # Threshold for stabilized predictions
        
        return delta_risk < threshold and sp > sp_threshold
    
    def run_active_learning(self, n_rounds=10, n_samples_per_round=10):
        """Run the active learning process."""
        for al_round in range(n_rounds):
            print(f"Active Learning Round {al_round+1}/{n_rounds}")
            
            # Get current labeled data
            X_labeled, y_labeled = self.get_labeled_data()
            
            # Train model on labeled data
            if len(X_labeled) > 0 and y_labeled is not None:
                self.model.fit(X_labeled, y_labeled)
            
            # Perform active testing if it's time
            if al_round % self.test_frequency == 0:
                print(f"  Performing active quiz...")
                self.perform_active_quiz(al_round)
                
                # Check for early stopping
                if self.check_early_stopping():
                    print(f"  Early stopping criteria met at round {al_round+1}")
                    break
            
            # Select samples for active learning
            X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
            if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
                print("  No more unlabeled data available.")
                break
                
            # 修正: X_unlabeled に対して選択を行い、実際のインデックスに変換
            selected_indices = self.active_learner.select_samples(
                X_unlabeled, n_samples=n_samples_per_round, 
                excluded_indices=[]  # X_unlabeled は既に除外済みなので空リスト
            )
            
            # 選択されたインデックスを元のプールのインデックスに変換
            original_indices = [unlabeled_indices[i] for i in selected_indices if i < len(unlabeled_indices)]
            
            # 選択されたサンプルがない場合の処理
            if not original_indices:
                print("  No more informative samples available.")
                break
                
            # Update labeled indices
            self.labeled_indices.extend(original_indices)
            self.active_learner.labeled_indices.extend(original_indices)
            
            # Print current stats
            if self.y_pool is not None and len(self.integrated_risk_history) > 0:
                print(f"  Labeled samples: {len(self.labeled_indices)}")
                print(f"  Test samples: {len(self.test_indices)}")
                print(f"  Feedback samples: {len(self.feedback_indices)}")
                print(f"  Estimated risk: {self.integrated_risk_history[-1]:.4f}")
                if len(self.true_risk_history) > 0:
                    print(f"  True risk: {self.true_risk_history[-1]:.4f}")
                    print(f"  Estimation error: {abs(self.integrated_risk_history[-1] - self.true_risk_history[-1]):.4f}")
        
        return self.model