import numpy as np
import torch # type: ignore

def entropy(probs):
    log_probs = np.log(probs + 1e-10) # probs=0の時-infに発散
    return -np.sum(probs * log_probs, axis=1) #行ごとに計算

# utils/losses.py
def cross_entropy_loss(y_pred, y_true):
    """
    クロスエントロピー損失を計算します。
    数値安定性を確保するために、クリッピングを適用します。
    
    Parameters:
    -----------
    y_pred : numpy.ndarray
        予測確率
    y_true : numpy.ndarray
        真のラベル
    
    Returns:
    --------
    loss : float
        クロスエントロピー損失
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # 数値安定性のためのクリッピング
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    
    if len(y_pred.shape) == 1:
        # 二値分類
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        # 多クラス分類
        if len(y_true.shape) == 1:
            # one-hotエンコーディングに変換
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((len(y_true), n_classes))
            y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
            y_true = y_true_one_hot
        
        # クロスエントロピー損失を計算
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
# y_true = [2,0,1]
# y_true_one_hot = [[1,0,0],
#                   [0,0,0],
#                   [0,0,1]]
# y_pred = [[0.1, 0.2, 0.7],
#           [0.8, 0.1, 0.1],  
#           [0.2, 0.6, 0.2]] 

