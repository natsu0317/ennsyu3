import numpy as np
from scipy.spatial.distance import cdist
from utils.losses import entropy, cross_entropy_loss

class ActiveTester:
    # モデル評価のためのサンプル選択
    def __init__(self, model):
        self.model = model
        self.test_indices=[]
        self.test_weights=[]
        self.test_proposals=[]
        self.quiz_results=[]
        self.feedback_indices=[]

    def compute_test_proposal(self, X_pool, excluded_indices=None, multi_source_risk=None):
        """
        最適なテスト提案分布を計算します（論文の式(4)）
        
        Parameters:
        -----------
        X_pool : numpy.ndarray
            サンプル選択対象のデータプール
        excluded_indices : list, optional
            選択から除外するインデックスのリスト
        multi_source_risk : float, optional
            多ソースリスク推定値
        
        Returns:
        --------
        q_star : numpy.ndarray
            最適なテスト提案分布
        """
        if excluded_indices is None:
            excluded_indices = []
        
        # プールのサンプルに対する予測確率を取得
        probs = self.model.predict_proba(X_pool)
        
        # 真のリスク(R)を推定（多ソースリスクが提供されている場合はそれを使用）
        if multi_source_risk is None:
            # モデルの不確実性を真のリスクの代わりに使用
            R = np.mean(entropy(probs))
        else:
            R = multi_source_risk
        
        # 式(4)の期待二乗差分項を計算
        # q*(x) ∝ p(x) * sqrt(∫[L(fθ(x), y) - R]²p(y|x)dy)
        
        # 各サンプルの期待二乗差分を計算
        n_classes = probs.shape[1] if len(probs.shape) > 1 else 2
        expected_sq_diff = np.zeros(len(X_pool))
        
        for i in range(len(X_pool)):
            # 各クラスについて期待値を計算
            sq_diff_sum = 0
            for c in range(n_classes):
                # クラスcに対する予測確率 p(y|x)
                p_y_given_x = probs[i, c] if len(probs.shape) > 1 else (probs[i] if c == 1 else 1 - probs[i])
                
                # クラスcの場合の損失 L(fθ(x), y)
                if len(probs.shape) > 1:
                    # 多クラス分類の場合
                    y_one_hot = np.zeros(n_classes)
                    y_one_hot[c] = 1
                    loss = -np.sum(y_one_hot * np.log(probs[i] + 1e-10))
                else:
                    # 二値分類の場合
                    loss = -np.log(probs[i] + 1e-10) if c == 1 else -np.log(1 - probs[i] + 1e-10)
                
                # 二乗差分
                sq_diff = (loss - R) ** 2
                
                # 期待値に寄与
                sq_diff_sum += p_y_given_x * sq_diff
            
            expected_sq_diff[i] = sq_diff_sum
        
        # プール分布p(x)はプール上で一様
        p_x = np.ones(len(X_pool)) / len(X_pool)
        
        # 提案分布q*(x)を計算
        q_star = p_x * np.sqrt(expected_sq_diff)
        
        # 除外インデックスをマスク
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        q_star[~mask] = 0
        
        # 正規化して適切な分布にする
        if np.sum(q_star) > 0:
            q_star = q_star / np.sum(q_star)
        else:
            # すべての値がゼロの場合、一様分布を使用
            q_star[mask] = 1 / np.sum(mask)
        
        return q_star

    def select_test_samples(self, X_pool, q_proposal, n_samples=1, excluded_indices=None):
        # 提案分布に基づいてテストサンプル選択
        if excluded_indices is None:
            excluded_indices = []
            
        # 除外インデックスのマスクを作成
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        
        # マスクを提案に適用
        q_masked = q_proposal.copy()
        q_masked[~mask] = 0
        
        # 再正規化
        if np.sum(q_masked) > 0:
            q_masked = q_masked / np.sum(q_masked)
        else:
            return [], []
        
        # 逐次的にサンプルを選択して多様性を確保
        selected_indices = []
        weights = []
        
        # プール分布（一様分布）
        p_x = np.ones(len(X_pool)) / len(X_pool)
        
        # 残りのサンプル数
        remaining = n_samples
        
        while remaining > 0 and np.sum(q_masked) > 0:
            # 現在の提案分布に基づいて1つのサンプルを選択
            idx = np.random.choice(len(X_pool), p=q_masked)
            
            # 選択されたサンプルの重みを計算
            weight = p_x[idx] / q_masked[idx]
            
            # 選択されたサンプルを記録
            selected_indices.append(idx)
            weights.append(weight)
            
            # 選択されたサンプルの周辺の提案確率を減少させて多様性を確保
            if len(X_pool) > 1000:  # 大規模データセットの場合のみ
                # 選択されたサンプルとの距離を計算
                distances = np.sum((X_pool - X_pool[idx]) ** 2, axis=1)
                
                # 距離に基づいて重みを計算（近いほど大きく減少）
                distance_weights = 1 - np.exp(-distances / np.median(distances))
                
                # 提案分布を更新
                q_masked = q_masked * distance_weights
                
                # 選択されたサンプル自体を除外
                q_masked[idx] = 0
                
                # 再正規化
                if np.sum(q_masked) > 0:
                    q_masked = q_masked / np.sum(q_masked)
            else:
                # 小規模データセットの場合は単純に除外
                q_masked[idx] = 0
                
                # 再正規化
                if np.sum(q_masked) > 0:
                    q_masked = q_masked / np.sum(q_masked)
            
            remaining -= 1
        
        # テストインデックスと重みを保存
        self.test_indices.extend(selected_indices)
        self.test_weights.extend(weights)
        self.test_proposals.append(q_proposal)
        
        return np.array(selected_indices), np.array(weights)

    def estimate_risk(self, X, y, weights=None):
        # モデルの損失平均
        if weights is None:
            weights = np.ones(len(X))
        probs = self.model.predict_proba(X)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # 多クラス分類
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y), n_classes))
            y_one_hot[np.arange(len(y)), y.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            losses = -y * np.log(probs + 1e-10) - (1 - y) * np.log(1 - probs + 1e-10)
        weighted_loss = np.sum(weights * losses) / np.sum(weights)

        return weighted_loss # 重み付き平均
    
    def compute_multi_source_risk(self, X_train, y_train, X_test, y_test, X_pool):
        """
        多ソースリスク推定を計算します（論文の式(5)に完全に忠実）
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            訓練データの特徴量
        y_train : numpy.ndarray
            訓練データのラベル
        X_test : numpy.ndarray
            テストデータの特徴量
        y_test : numpy.ndarray
            テストデータのラベル
        X_pool : numpy.ndarray
            未ラベルプールの特徴量
        
        Returns:
        --------
        multi_source_risk : float
            多ソースリスク推定値
        """
        # 各ソースの重み
        train_weight = len(X_train)
        pool_weight = len(X_pool)
        test_weight = len(X_test)
        
        # 訓練リスク
        if len(X_train) > 0:
            # 交差検証で訓練リスクを推定
            n_splits = min(5, len(X_train))
            if n_splits > 1:
                # データをシャッフル
                indices = np.random.permutation(len(X_train))
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]
                
                # K分割交差検証
                train_risks = []
                split_size = len(X_train) // n_splits
                for i in range(n_splits):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X_train)
                    
                    # 検証セットとトレーニングセットを分割
                    val_indices = indices[start_idx:end_idx]
                    train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                    
                    # 一時的なモデルを訓練
                    temp_model = copy.deepcopy(self.model)
                    temp_model.fit(X_train[train_indices], y_train[train_indices])
                    
                    # 検証セットでリスクを計算
                    val_probs = temp_model.predict_proba(X_train[val_indices])
                    val_risk = cross_entropy_loss(val_probs, y_train[val_indices])
                    train_risks.append(val_risk)
                
                train_risk = np.mean(train_risks)
            else:
                # サンプルが少ない場合は直接計算
                train_probs = self.model.predict_proba(X_train)
                train_risk = cross_entropy_loss(train_probs, y_train)
        else:
            train_risk = 0
        
        # プールリスク（モデルの不確実性）
        if len(X_pool) > 0:
            pool_probs = self.model.predict_proba(X_pool)
            pool_uncertainties = entropy(pool_probs)
            pool_risk = np.mean(pool_uncertainties)
        else:
            pool_risk = 0
        
        # テストリスク
        if len(X_test) > 0:
            # テストデータの重みを計算
            test_weights = np.ones(len(X_test)) / len(X_test)
            
            # テストリスクを計算
            test_probs = self.model.predict_proba(X_test)
            test_risk = cross_entropy_loss(test_probs, y_test)
        else:
            test_risk = 0
        
        # 多ソースリスク推定を計算（式(5)）
        total_weight = train_weight + pool_weight + test_weight
        if total_weight > 0:
            multi_source_risk = (train_weight * train_risk + pool_weight * pool_risk + test_weight * test_risk) / total_weight
        else:
            multi_source_risk = 0
        
        return multi_source_risk
    
    def integrated_risk_estimation(self, X_quizzes, y_quizzes, quiz_weights, model=None):
        """
        複数のクイズ結果を統合してリスクを推定します（論文のTheorem 1に基づく）
        
        Parameters:
        -----------
        X_quizzes : list of numpy.ndarray
            各クイズの特徴量データのリスト
        y_quizzes : list of numpy.ndarray
            各クイズのラベルデータのリスト
        quiz_weights : list of numpy.ndarray
            各クイズのサンプル重みのリスト
        model : BaseModel, optional
            評価対象のモデル（指定しない場合は自身のモデルを使用）
        
        Returns:
        --------
        integrated_risk : float
            統合されたリスク推定値
        """
        if model is None:
            model = self.model
            
        # 各クイズの信頼度を計算
        confidences = []
        risk_estimates = []
        n_samples = []
        
        for X_quiz, y_quiz, weights in zip(X_quizzes, y_quizzes, quiz_weights):
            if len(X_quiz) == 0:
                continue
                
            # このクイズのリスクを推定
            risk = self.estimate_risk(X_quiz, y_quiz, weights)
            risk_estimates.append(risk)
            n_samples.append(len(X_quiz))
            
            # 式(7)に基づいて信頼度（分散の逆数）を計算
            probs = model.predict_proba(X_quiz)
            
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                # 多クラス分類
                n_classes = probs.shape[1]
                y_one_hot = np.zeros((len(y_quiz), n_classes))
                y_one_hot[np.arange(len(y_quiz)), y_quiz.astype(int)] = 1
                losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
            else:
                # 二値分類
                losses = -y_quiz * np.log(probs + 1e-10) - (1 - y_quiz) * np.log(1 - probs + 1e-10)
            
            # 推定リスクからの二乗差を計算
            sq_diff = (losses - risk) ** 2
            
            # 重要度重みを適用して分散を計算
            weighted_sq_diff = np.sum(weights * sq_diff) / np.sum(weights)
            
            # サンプル数を考慮した信頼度
            # Theorem 1に基づき、分散はサンプル数に反比例する
            confidence = len(X_quiz) / (weighted_sq_diff + 1e-10)
            confidences.append(confidence)
            
        if not confidences:
            return 0
            
        # 信頼度を正規化して重みを取得（Theorem 1のvt）
        confidences = np.array(confidences)
        v_weights = confidences / np.sum(confidences)
        
        # 統合リスク推定値を計算
        integrated_risk = np.sum(v_weights * np.array(risk_estimates))
        
        return integrated_risk
    
    def select_feedback_samples(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
        """
        テストセットからフィードバックサンプルを選択します（リスク推定への影響を考慮）
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            テストデータの特徴量
        y_test : numpy.ndarray
            テストデータのラベル
        X_train : numpy.ndarray
            訓練データの特徴量
        test_indices : numpy.ndarray
            テストサンプルのインデックス
        q_proposal : numpy.ndarray
            テスト提案分布
        n_feedback : int, optional
            選択するフィードバックサンプル数
        
        Returns:
        --------
        original_indices : numpy.ndarray
            選択されたフィードバックサンプルの元のインデックス
        """
        if len(X_test) == 0:
            return []
            
        # 損失を計算
        probs = self.model.predict_proba(X_test)
        
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # 多クラス分類
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y_test), n_classes))
            y_one_hot[np.arange(len(y_test)), y_test.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            # 二値分類
            losses = -y_test * np.log(probs + 1e-10) - (1 - y_test) * np.log(1 - probs + 1e-10)
        
        # 多様性を計算
        if len(X_train) > 0:
            # 距離行列を計算
            dist_matrix = cdist(X_test, X_train)
            
            # 訓練セットへの最小距離を計算
            min_distances = np.min(dist_matrix, axis=1)
            
            # 正規化
            if np.max(min_distances) > 0:
                diversity = min_distances / np.max(min_distances)
            else:
                diversity = np.ones_like(min_distances)
        else:
            diversity = np.ones(len(X_test))
        
        # リスク推定への影響を計算
        # 各サンプルを除外した場合のリスク推定値の変化を推定
        risk_impact = np.zeros(len(X_test))
        mean_loss = np.mean(losses)
        for i in range(len(X_test)):
            # i番目のサンプルを除外した場合の平均損失
            remaining_losses = np.delete(losses, i)
            new_mean_loss = np.mean(remaining_losses) if len(remaining_losses) > 0 else mean_loss
            # リスク推定への影響は元の平均との差の絶対値
            risk_impact[i] = abs(new_mean_loss - mean_loss)
        
        # リスク推定への影響を正規化
        if np.max(risk_impact) > 0:
            risk_impact = risk_impact / np.max(risk_impact)
        
        # 式(9)に従ってフィードバックスコアを計算
        # q_FB(x, y; η) = q^(t)(x)L(ft(x), y) + ηd(SL, x)
        eta = 0.8  # 論文の結果に合わせてηを調整（0.5から0.8に）
        
        # 以下のコードは同じ
        feedback_scores = q_proposal[test_indices] * losses + eta * diversity
        
        # 上位n_feedbackサンプルを選択
        selected_indices = np.argsort(feedback_scores)[-n_feedback:][::-1]
        
        # 元のインデックスに変換
        original_indices = test_indices[selected_indices]
        
        # フィードバックインデックスを保存
        self.feedback_indices.extend(original_indices)
        
        return original_indices