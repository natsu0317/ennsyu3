import numpy as np
from scipy.spatial.distance import cdist
from utils.losses import entropy, cross_entropy_loss

class ActiveTester:
    """Active tester for model evaluation."""
    def __init__(self, model):
        self.model = model
        self.test_indices = []
        self.test_weights = []
        self.test_proposals = []
        self.quiz_results = []
        self.feedback_indices = []
        
    def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
        """Compute the optimal test proposal distribution."""
        if excluded_indices is None:
            excluded_indices = []
            
        # Get predictions for the pool
        probs = self.model.predict_proba(X_pool)
        
        # Estimate the true risk (R) using multi-source risk if provided
        if multi_source_risk is None:
            # Use model uncertainty as a proxy for true risk
            R = np.mean(entropy(probs))
        else:
            R = multi_source_risk
            
        # Calculate the expected squared difference term in equation (4)
        # q*(x) ∝ p(x) * sqrt(∫[L(fθ(x), y) - R]²p(y|x)dy)
        
        # For classification with cross-entropy loss, this is related to the uncertainty
        # We'll use entropy as a proxy for the integral term
        uncertainty = entropy(probs)
        
        # The pool distribution p(x) is uniform over the pool
        p_x = np.ones(len(X_pool)) / len(X_pool)
        
        # Calculate the proposal q*(x)
        q_star = p_x * np.sqrt(np.abs(uncertainty - R))
        
        # Mask out excluded indices
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        q_star[~mask] = 0
        
        # Normalize to get a proper distribution
        if np.sum(q_star) > 0:
            q_star = q_star / np.sum(q_star)
        else:
            # If all values are zero, use uniform distribution
            q_star[mask] = 1 / np.sum(mask)
            
        return q_star
    
    def select_test_samples(self, X_pool, q_proposal, n_samples=1, excluded_indices=None):
        """Select test samples based on the proposal distribution."""
        if excluded_indices is None:
            excluded_indices = []
            
        # Create a mask for excluded indices
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        
        # Apply the mask to the proposal
        q_masked = q_proposal.copy()
        q_masked[~mask] = 0
        
        # Renormalize
        if np.sum(q_masked) > 0:
            q_masked = q_masked / np.sum(q_masked)
        else:
            return [], []
        
        # Sample indices based on the proposal distribution
        selected_indices = np.random.choice(
            np.arange(len(X_pool)), 
            size=min(n_samples, np.sum(mask)), 
            replace=False, 
            p=q_masked
        )
        
        # Calculate importance weights for the selected samples
        p_x = np.ones(len(X_pool)) / len(X_pool)  # Uniform pool distribution
        weights = p_x[selected_indices] / q_masked[selected_indices]
        
        self.test_indices.extend(selected_indices)
        self.test_weights.extend(weights)
        self.test_proposals.append(q_masked)
        
        return selected_indices, weights
    
    def estimate_risk(self, X, y, weights=None):
        """Estimate the risk using importance weighted samples."""
        if weights is None:
            weights = np.ones(len(X))
            
        probs = self.model.predict_proba(X)
        
        # For classification, we use cross-entropy loss
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # Multi-class classification
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y), n_classes))
            y_one_hot[np.arange(len(y)), y.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            # Binary classification
            losses = -y * np.log(probs + 1e-10) - (1 - y) * np.log(1 - probs + 1e-10)
            
        # Apply importance weights
        weighted_loss = np.sum(weights * losses) / np.sum(weights)
        
        return weighted_loss
    
    def compute_multi_source_risk(self, X_train, y_train, X_test, y_test, X_pool, test_weights=None):
        """Compute the multi-source risk estimate as in equation (5)."""
        # Training risk
        train_risk = cross_entropy_loss(self.model.predict_proba(X_train), y_train)
        
        # Model uncertainty risk (over unlabeled pool)
        pool_probs = self.model.predict_proba(X_pool)
        pool_risk = np.mean(entropy(pool_probs))
        
        # Test risk
        if len(X_test) > 0:
            test_risk = self.estimate_risk(X_test, y_test, weights=test_weights)
        else:
            test_risk = 0
            
        # Combine the risks according to equation (5)
        multi_source_risk = (len(X_train) * train_risk + len(X_pool) * pool_risk + len(X_test) * test_risk) / \
                           (len(X_train) + len(X_pool) + len(X_test))
        
        return multi_source_risk
    
    def integrated_risk_estimation(self, X_quizzes, y_quizzes, quiz_weights, model=None):
        """Compute the integrated risk estimation from multiple quizzes."""
        if model is None:
            model = self.model
            
        # Calculate confidence for each quiz
        confidences = []
        risk_estimates = []
        
        for X_quiz, y_quiz, weights in zip(X_quizzes, y_quizzes, quiz_weights):
            if len(X_quiz) == 0:
                continue
                
            # Estimate risk for this quiz
            risk = self.estimate_risk(X_quiz, y_quiz, weights)
            risk_estimates.append(risk)
            
            # Calculate confidence (inverse of variance)
            probs = model.predict_proba(X_quiz)
            
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                # Multi-class classification
                n_classes = probs.shape[1]
                y_one_hot = np.zeros((len(y_quiz), n_classes))
                y_one_hot[np.arange(len(y_quiz)), y_quiz.astype(int)] = 1
                losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
            else:
                # Binary classification
                losses = -y_quiz * np.log(probs + 1e-10) - (1 - y_quiz) * np.log(1 - probs + 1e-10)
            
            # Calculate squared difference from the estimated risk
            sq_diff = (losses - risk) ** 2
            
            # Apply importance weights
            weighted_sq_diff = np.sum(weights * sq_diff) / np.sum(weights)
            
            # Confidence is inverse of variance
            confidence = 1 / (weighted_sq_diff + 1e-10)
            confidences.append(confidence)
            
        if not confidences:
            return 0
            
        # Normalize confidences to get weights
        confidences = np.array(confidences)
        v_weights = confidences / np.sum(confidences)
        
        # Compute integrated risk estimate
        integrated_risk = np.sum(v_weights * np.array(risk_estimates))
        
        return integrated_risk
    
    def select_feedback_samples(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
        """Select feedback samples from the test set."""
        if len(X_test) == 0:
            return []
            
        # Compute losses
        probs = self.model.predict_proba(X_test)
        
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # Multi-class classification
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y_test), n_classes))
            y_one_hot[np.arange(len(y_test)), y_test.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            # Binary classification
            losses = -y_test * np.log(probs + 1e-10) - (1 - y_test) * np.log(1 - probs + 1e-10)
        
        # Compute diversity
        if len(X_train) > 0:
            # Compute distance matrix
            dist_matrix = cdist(X_test, X_train)
            
            # Compute minimum distance to training set
            min_distances = np.min(dist_matrix, axis=1)
            
            # Normalize
            if np.max(min_distances) > 0:
                diversity = min_distances / np.max(min_distances)
            else:
                diversity = np.ones_like(min_distances)
        else:
            diversity = np.ones(len(X_test))
            
        # Compute feedback score according to equation (9)
        # q_FB(x, y) = q^(t)(x) * L(f_t(x), y) + η * d(S_L, x)
        eta = 0.5  # Scaling parameter
        feedback_scores = q_proposal[test_indices] * losses + eta * diversity
        
        # Select top n_feedback samples
        selected_indices = np.argsort(feedback_scores)[-n_feedback:][::-1]
        
        # Convert to original indices
        original_indices = test_indices[selected_indices]
        
        self.feedback_indices.extend(original_indices)
        
        return original_indices