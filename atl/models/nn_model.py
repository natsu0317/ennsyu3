import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel

class MLP(nn.Module):
    """
    MNIST/FashionMNIST用のシンプルなMLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # フラット化
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    """
    CIFAR10用のシンプルなCNN
    論文の付録C.2.1に基づく
    """
    def __init__(self, output_dim):
        super().__init__()
        # 論文の付録C.2.1に基づいたCIFAR10用のCNNアーキテクチャ
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # CIFAR10は32x32なので、2回プーリングすると8x8になる
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 入力形状を(N, C, H, W)に変換
        if len(x.shape) == 2:
            # フラット化された入力の場合
            x = x.view(-1, 3, 32, 32)
        elif len(x.shape) == 4 and x.shape[1] != 3:
            # (N, H, W, C)形式の場合、(N, C, H, W)に変換
            x = x.permute(0, 3, 1, 2)
            
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleNN(nn.Module):
    """
    付録Cの実験設定に基づく単純なニューラルネットワークモデル
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        # 付録Cで言及されているアーキテクチャに基づいて層を定義
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)  # ドロップアウトを追加
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)  # ドロップアウトを追加
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # 重みの初期化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        # フラット化
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class NNModel(BaseModel):
    """
    ニューラルネットワークモデルのラッパークラス
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dataset_name="mnist", lr=0.001, batch_size=64, epochs=5):
        super().__init__()
        # データセットに応じたモデルを選択
        if dataset_name in ["mnist", "fashion_mnist"]:
            # 論文の付録C.2.1に基づくと、MNISTとFashionMNISTにはMLPを使用
            self.model = SimpleNN(input_dim, hidden_dim, output_dim)
        else:  # cifar10
            # 論文の付録C.2.1に基づくと、CIFAR10にはCNNを使用
            self.model = SimpleCNN(output_dim)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Using model on device: {self.device}")

    def fit(self, X, y):
        """
        データからモデルを学習します
        
        Parameters:
        -----------
        X : numpy.ndarray
            特徴量データ
        y : numpy.ndarray
            ラベルデータ
        
        Returns:
        --------
        self : NNModel
            学習済みモデル
        """
        self.model.train()
        
        # NumPy配列をPyTorchテンソルに変換
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # データセットとデータローダーを作成
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 指定されたエポック数だけ学習を実行
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                # 勾配をゼロにリセット
                self.optimizer.zero_grad()
                # 順伝播計算
                outputs = self.model(inputs)
                # 損失を計算
                loss = self.criterion(outputs, targets)
                # 逆伝播計算
                loss.backward()
                # パラメータを更新
                self.optimizer.step()
        
        return self
    
    def predict(self, X):
        """
        クラスラベルを予測します
        
        Parameters:
        -----------
        X : numpy.ndarray
            特徴量データ
        
        Returns:
        --------
        y_pred : numpy.ndarray
            予測されたクラスラベル
        """
        self.model.eval()
        
        # NumPy配列をPyTorchテンソルに変換
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 勾配計算を無効化
        with torch.no_grad():
            # 順伝播計算
            outputs = self.model(X_tensor)
            # 最大値のインデックス（予測クラス）を取得
            _, predicted = torch.max(outputs, 1)
        
        # PyTorchテンソルをNumPy配列に変換して返す
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        クラス確率を予測します
        
        Parameters:
        -----------
        X : numpy.ndarray
            特徴量データ
        
        Returns:
        --------
        probs : numpy.ndarray
            予測されたクラス確率
        """
        self.model.eval()
        
        # NumPy配列をPyTorchテンソルに変換
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 勾配計算を無効化
        with torch.no_grad():
            # 順伝播計算
            outputs = self.model(X_tensor)
            # ソフトマックス関数を適用して確率に変換
            probs = F.softmax(outputs, dim=1)
        
        # PyTorchテンソルをNumPy配列に変換して返す
        return probs.cpu().numpy()