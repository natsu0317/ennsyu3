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
    """
    アクティブラーニング用にデータを準備します（付録Cの実験設定に基づく）
    
    Parameters:
    -----------
    train_dataset : Dataset
        訓練データセット
    test_dataset : Dataset
        テストデータセット
    n_initial : int, optional
        初期ラベル付きデータのサイズ
    n_pool : int, optional
        プールのサイズ
    n_holdout : int, optional
        ホールドアウトセットのサイズ
    
    Returns:
    --------
    X_initial : numpy.ndarray
        初期ラベル付きデータの特徴量
    y_initial : numpy.ndarray
        初期ラベル付きデータのラベル
    X_pool : numpy.ndarray
        プールの特徴量
    y_pool : numpy.ndarray
        プールのラベル
    X_holdout : numpy.ndarray
        ホールドアウトセットの特徴量
    y_holdout : numpy.ndarray
        ホールドアウトセットのラベル
    """
    # データを結合
    X_all = np.vstack([train_dataset.data, test_dataset.data])
    y_all = np.hstack([train_dataset.targets, test_dataset.targets])
    
    # 必要に応じてデータを整形
    if len(X_all.shape) > 2:
        X_all = X_all.reshape(X_all.shape[0], -1)
    
    # 付録Cの実験設定に基づいてデータを前処理
    if X_all.shape[1] == 784:  # MNISTまたはFashionMNIST
        # [0, 1]の範囲に正規化
        X_all = X_all / 255.0
    else:  # CIFAR10
        # [-1, 1]の範囲に正規化
        X_all = (X_all / 127.5) - 1.0
    
    # クラスバランスを考慮してデータを分割
    indices_by_class = [np.where(y_all == c)[0] for c in range(np.max(y_all) + 1)]
    
    initial_indices = []
    pool_indices = []
    holdout_indices = []
    
    # 各クラスから均等にサンプルを選択
    n_classes = len(indices_by_class)
    n_initial_per_class = n_initial // n_classes
    n_pool_per_class = n_pool // n_classes
    n_holdout_per_class = n_holdout // n_classes
    
    for class_indices in indices_by_class:
        np.random.shuffle(class_indices)
        initial_indices.extend(class_indices[:n_initial_per_class])
        pool_indices.extend(class_indices[n_initial_per_class:n_initial_per_class+n_pool_per_class])
        holdout_indices.extend(class_indices[n_initial_per_class+n_pool_per_class:n_initial_per_class+n_pool_per_class+n_holdout_per_class])
    
    # インデックスをシャッフル
    np.random.shuffle(initial_indices)
    np.random.shuffle(pool_indices)
    np.random.shuffle(holdout_indices)
    
    X_initial = X_all[initial_indices]
    y_initial = y_all[initial_indices]
    
    X_pool = X_all[pool_indices]
    y_pool = y_all[pool_indices]
    
    X_holdout = X_all[holdout_indices]
    y_holdout = y_all[holdout_indices]
    
    return X_initial, y_initial, X_pool, y_pool, X_holdout, y_holdout