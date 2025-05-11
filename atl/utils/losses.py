import numpy as np
import torch # type: ignore

def entropy(probs):
    log_probs = np.log(probs + 1e-10) # probs=0の時-infに発散
    return -np.sum(probs * log_probs, axis=1) #行ごとに計算

def cross_entropy_loss(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if len(y_pred.shape) == 1:
        return -np.mean(y_true * np.log(y_pred+1e-10) + (1-y_true) * np.log(1-y_pred+1e-10))
    else:
        # 多クラス分類
        if len(y_true.shape) == 1:
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((len(y_true), n_classes))
            y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_one_hot

        return -np.mean(np.sum(y_true * np.log(y_pred+1e-10), axis=1))
    
# y_true = [2,0,1]
# y_true_one_hot = [[1,0,0],
#                   [0,0,0],
#                   [0,0,1]]
# y_pred = [[0.1, 0.2, 0.7],
#           [0.8, 0.1, 0.1],  
#           [0.2, 0.6, 0.2]] 

