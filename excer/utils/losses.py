import numpy as np
import torch # type: ignore

def entropy(probs):
    """Calculate entropy of probability distributions."""
    log_probs = np.log(probs + 1e-10)
    return -np.sum(probs * log_probs, axis=1)

def cross_entropy_loss(y_pred, y_true):
    """Calculate cross entropy loss."""
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    if len(y_pred.shape) == 1:
        # Binary classification
        return -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
    else:
        # Multi-class classification
        if len(y_true.shape) == 1:
            # Convert to one-hot
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((len(y_true), n_classes))
            y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_one_hot
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))