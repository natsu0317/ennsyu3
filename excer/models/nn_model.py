import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
from models.base_model import BaseModel

class SimpleNN(nn.Module):
    """Simple neural network for classification."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NNModel(BaseModel):
    """Neural Network model wrapper."""
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, batch_size=32, epochs=10):
        super().__init__()
        self.model = SimpleNN(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dim = output_dim
        
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
            
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.numpy()
    
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probs = F.softmax(outputs, dim=1)
            
        return probs.numpy()