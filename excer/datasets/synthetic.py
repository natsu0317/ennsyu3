import numpy as np
from sklearn.datasets import make_moons

def create_moon_dataset_with_clusters():
    """Create a moon-shaped dataset with two smaller inverted moon-shaped clusters."""
    # Generate the main moon dataset
    X_main, y_main = make_moons(n_samples=2000, noise=0.1, random_state=42)
    
    # Generate two smaller inverted moon clusters
    X_cluster1, y_cluster1 = make_moons(n_samples=250, noise=0.05, random_state=43)
    X_cluster1 = X_cluster1 * 0.5 - np.array([2.0, 2.0])  # Scale and shift
    y_cluster1 = 1 - y_cluster1  # Invert labels
    
    X_cluster2, y_cluster2 = make_moons(n_samples=250, noise=0.05, random_state=44)
    X_cluster2 = X_cluster2 * 0.5 + np.array([2.0, 2.0])  # Scale and shift
    y_cluster2 = 1 - y_cluster2  # Invert labels
    
    # Combine datasets
    X = np.vstack([X_main, X_cluster1, X_cluster2])
    y = np.hstack([y_main, y_cluster1, y_cluster2])
    
    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y