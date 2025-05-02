import numpy as np
import torch # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import Dataset # type: ignore

class ImageDatasetWrapper(Dataset):
    """Wrapper for image datasets to support active learning."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = []
        self.targets = []
        
        # Extract data and targets
        for i in range(len(dataset)):
            img, target = dataset[i]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            self.data.append(img)
            self.targets.append(target)
            
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_dataset(name):
    """Load a dataset by name."""
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return ImageDatasetWrapper(train_dataset), ImageDatasetWrapper(test_dataset)

def prepare_data_for_al(train_dataset, test_dataset, n_initial=500, n_pool=30000, n_holdout=10000):
    """Prepare data for active learning."""
    # Combine train and test datasets
    X_all = np.vstack([train_dataset.data, test_dataset.data])
    y_all = np.hstack([train_dataset.targets, test_dataset.targets])
    
    # Reshape data if needed
    if len(X_all.shape) > 2:
        X_all = X_all.reshape(X_all.shape[0], -1)
    
    # Split into initial, pool, and holdout sets
    indices = np.random.permutation(len(X_all))
    initial_indices = indices[:n_initial]
    pool_indices = indices[n_initial:n_initial+n_pool]
    holdout_indices = indices[n_initial+n_pool:n_initial+n_pool+n_holdout]
    
    X_initial = X_all[initial_indices]
    y_initial = y_all[initial_indices]
    
    X_pool = X_all[pool_indices]
    y_pool = y_all[pool_indices]
    
    X_holdout = X_all[holdout_indices]
    y_holdout = y_all[holdout_indices]
    
    return X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout