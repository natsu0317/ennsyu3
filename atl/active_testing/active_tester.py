import numpy as np
from scipy.spatial.distance import cdist
from utils.losses import entropy, cross_entropy_loss
import copy

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
        最適なテスト提案分布q*(x)を計算します（論文の式(4)）
        
        Parameters:
        -----------
        X_pool : numpy.ndarray
            サンプル選択対象のデータプール
        excluded_indices : list, optional
            選択から除外するインデックスのリスト
        multi_source_risk : float, optional
            多ソースリスク推定値R
        
        Returns:
        --------
        q_star : numpy.ndarray
            最適なテスト提案分布q*(x)
        """
        if excluded_indices is None:
            excluded_indices = []
            
        # プールのサンプルに対する予測確率を取得
        probs = self.model.predict_proba(X_pool)
        
        # 真のリスクR（論文の記号に合わせる）
        if multi_source_risk is None:
            # モデルの不確実性を真のリスクの代わりに使用
            R = np.mean(entropy(probs))
        else:
            R = multi_source_risk
            
        # 式(4)の期待二乗差分項を計算
        # q*(x) ∝ p(x) * sqrt(∫[L(fθ(x), y) - R]²p(y|x)dy)
        
        # 各クラスの予測確率に基づいて、損失の期待二乗差を計算
        expected_sq_diff = np.zeros(len(X_pool))
        
        # 多クラス分類の場合
        if probs.shape[1] > 1:
            n_classes = probs.shape[1]
            for c in range(n_classes):
                # クラスcの場合の損失
                y_c = np.zeros(n_classes)
                y_c[c] = 1
                loss_c = -np.sum(y_c * np.log(probs + 1e-10), axis=1)
                
                # クラスcの確率で重み付け
                expected_sq_diff += probs[:, c] * (loss_c - R) ** 2
        else:
            # 二値分類の場合
            # クラス1の場合の損失
            loss_1 = -np.log(probs + 1e-10)
            # クラス0の場合の損失
            loss_0 = -np.log(1 - probs + 1e-10)
            
            # 各クラスの確率で重み付け
            expected_sq_diff += probs[:, 0] * (loss_1 - R) ** 2 + (1 - probs[:, 0]) * (loss_0 - R) ** 2
        
        # プール分布p(x)はプール上で一様
        p_x = np.ones(len(X_pool)) / len(X_pool)
        
        # 提案分布q*(x)を計算（論文の式(4)に従う）
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
        """
        提案分布に基づいてテストサンプルを選択します
        
        Parameters:
        -----------
        X_pool : numpy.ndarray
            サンプル選択対象のデータプール
        q_proposal : numpy.ndarray
            テスト提案分布
        n_samples : int, optional
            選択するサンプル数
        excluded_indices : list, optional
            選択から除外するインデックスのリスト
        
        Returns:
        --------
        selected_indices : numpy.ndarray
            選択されたサンプルのインデックス配列
        weights : numpy.ndarray
            選択されたサンプルの重み
        """
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
        
        # 提案分布に基づいてインデックスをサンプリング
        # より多くのサンプルを選択（論文の図3に近づけるため）
        actual_n_samples = min(n_samples, np.sum(mask))
        if actual_n_samples == 0:
            return [], []
            
        selected_indices = np.random.choice(
            np.arange(len(X_pool)), 
            size=actual_n_samples, 
            replace=False, 
            p=q_masked
        )
        
        # 選択されたサンプルの重要度重みを計算
        p_x = np.ones(len(X_pool)) / len(X_pool)  # 一様プール分布
        weights = p_x[selected_indices] / q_masked[selected_indices]
        
        # テストインデックスと重みを保存
        self.test_indices.extend(selected_indices)
        self.test_weights.extend(weights)
        self.test_proposals.append(q_masked)
        
        print(f"Selected {len(selected_indices)} test samples, total: {len(self.test_indices)}")
        
        return selected_indices, weights

    def estimate_risk(self, X, y, weights=None):
        """重み付きサンプルからリスクを推定します（論文の式(2)）"""
        if weights is None:
            weights = np.ones(len(X))
            
        # モデルの予測確率を取得
        probs = self.model.predict_proba(X)
        
        # 分類問題の場合、交差エントロピー損失を使用
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # 多クラス分類
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y), n_classes))
            y_one_hot[np.arange(len(y)), y.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            # 二値分類
            losses = -y * np.log(probs + 1e-10) - (1 - y) * np.log(1 - probs + 1e-10)
            
        # 重要度重みを適用
        weighted_loss = np.sum(weights * losses) / np.sum(weights)
        
        # デバッグ出力
        print(f"Estimate risk calculation:")
        print(f"  Average loss: {np.mean(losses)}")
        print(f"  Average weight: {np.mean(weights)}")
        print(f"  Weighted loss: {weighted_loss}")
        
        return weighted_loss
    
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
        """複数のクイズ結果を統合してリスクを推定します（論文の式(7)）"""
        if model is None:
            model = self.model
            
        # 各クイズの信頼度Ctを計算
        confidences = []  # Ct
        risk_estimates = []  # R̂t
        
        for t, (X_quiz, y_quiz, weights) in enumerate(zip(X_quizzes, y_quizzes, quiz_weights)):
            if len(X_quiz) == 0:
                continue
                    
            # このクイズのリスクを推定（R̂t）
            risk_t = self.estimate_risk(X_quiz, y_quiz, weights)
            risk_estimates.append(risk_t)
            
            # 信頼度（分散の逆数）を計算（Ct = 1/σ²t）
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
            sq_diff = (losses - risk_t) ** 2
            
            # 重要度重みを適用
            weighted_sq_diff = np.sum(weights * sq_diff) / np.sum(weights)
            
            # 信頼度Ctは分散の逆数
            confidence_t = 1 / (weighted_sq_diff + 1e-10)
            confidences.append(confidence_t)
            
        if not confidences:
            return 0
            
        # 信頼度を正規化して重みvtを取得
        confidences = np.array(confidences)
        v_weights = confidences / np.sum(confidences)
        
        # 統合リスク推定値R̃を計算
        integrated_risk = np.sum(v_weights * np.array(risk_estimates))
        
        # デバッグ出力
        print(f"Integrated risk calculation:")
        print(f"  Risk estimates: {risk_estimates}")
        print(f"  Confidences: {confidences}")
        print(f"  v_weights: {v_weights}")
        print(f"  Integrated risk: {integrated_risk}")
        
        return integrated_risk

    def select_feedback_samples(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
        """
        テストセットからフィードバックサンプルを選択します（論文の式(9)）
        
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
            テスト提案分布q^(t)(x)
        n_feedback : int, optional
            選択するフィードバックサンプル数
        
        Returns:
        --------
        original_indices : numpy.ndarray
            選択されたフィードバックサンプルの元のインデックス
        """
        if len(X_test) == 0 or len(test_indices) == 0:
            return []
            
        # デバッグ出力
        print(f"  Feedback selection - X_test shape: {X_test.shape}, test_indices length: {len(test_indices)}")
        
        # X_testとtest_indicesの長さが一致しない場合、最新のテストサンプルのみを使用
        if len(X_test) != len(test_indices):
            print(f"  Warning: X_test length ({len(X_test)}) != test_indices length ({len(test_indices)})")
            print(f"  Using only the latest test samples for feedback selection")
            X_test = X_test[-len(test_indices):]
            y_test = y_test[-len(test_indices):]
        
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
            # 論文の記述に従い、多様性メトリックd(SL, x)を計算
            # d(SL, x) = sqrt(x^T A_L^(-1) x), where A_L = ε*I + Σ_{z∈SL} z*z^T
            
            # A_L行列を計算
            epsilon = 1e-5  # 小さな正の値εを設定
            A_L = epsilon * np.eye(X_train.shape[1])
            
            for z in X_train:
                A_L += np.outer(z, z)
                
            # 各テストサンプルに対してd(SL, x)を計算
            diversity = np.zeros(len(X_test))
            
            try:
                A_L_inv = np.linalg.inv(A_L)
                
                for i, x in enumerate(X_test):
                    diversity[i] = np.sqrt(x.dot(A_L_inv).dot(x))
                    
                # 正規化
                if np.max(diversity) > 0:
                    diversity = diversity / np.max(diversity)
                else:
                    diversity = np.ones_like(diversity)
            except np.linalg.LinAlgError:
                # 行列が特異の場合、代替アプローチを使用
                print("Warning: Singular matrix in diversity calculation. Using distance-based diversity.")
                dist_matrix = cdist(X_test, X_train)
                min_distances = np.min(dist_matrix, axis=1)
                
                if np.max(min_distances) > 0:
                    diversity = min_distances / np.max(min_distances)
                else:
                    diversity = np.ones_like(min_distances)
        else:
            diversity = np.ones(len(X_test))
        
        # デバッグ出力
        print(f"  Feedback calculation - losses shape: {losses.shape}, diversity shape: {diversity.shape}, q_proposal[test_indices] shape: {q_proposal[test_indices].shape}")
        
        # 式(9)に従ってフィードバックスコアを計算
        # q_FB(x, y; η) = q^(t)(x) * L(f_t(x), y) + η * d(S_L, x)
        eta = 0.5  # スケーリングパラメータη
        feedback_scores = q_proposal[test_indices] * losses + eta * diversity
        
        # 上位n_feedbackサンプルを選択
        if n_feedback > len(test_indices):
            n_feedback = len(test_indices)
            print(f"  Warning: Requested more feedback samples than available. Using {n_feedback} samples.")
            
        selected_indices = np.argsort(feedback_scores)[-n_feedback:][::-1]
        
        # 元のインデックスに変換
        original_indices = test_indices[selected_indices]
        
        # フィードバックインデックスを保存
        self.feedback_indices.extend(original_indices)
        
        return original_indices