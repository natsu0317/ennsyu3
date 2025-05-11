import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title, labeled_indices=None, test_indices=None, feedback_indices=None):
    """Plot the decision boundary of the model."""
    plt.figure(figsize=(10, 8))
    
    # Define the grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on the grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    
    if Z.shape[1] > 1:  # Multi-class
        Z = Z[:, 1]
    
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    
    # Highlight labeled, test, and feedback points
    if labeled_indices is not None:
        plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], s=100, facecolors='none', edgecolors='g', linewidths=2, label='AL Sample')
    
    if test_indices is not None:
        plt.scatter(X[test_indices, 0], X[test_indices, 1], s=100, facecolors='none', edgecolors='b', linewidths=2, label='AT Sample')
    
    if feedback_indices is not None:
        plt.scatter(X[feedback_indices, 0], X[feedback_indices, 1], s=100, facecolors='none', edgecolors='r', linewidths=2, label='Feedback Sample')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    return plt