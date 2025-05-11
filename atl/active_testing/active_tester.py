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
        # テスト提案分布(式(4))
        if excluded_indices is None:
            excluded_indices = []
        probs = self.model.predict_proba(X_pool) # poolの予測確率
        if multi_source_risk is None:
            R = np.mean(entropy(probs))
        else:
            R = multi_source_risk
        uncertainty = entropy(probs)
        p_x = np.ones(len(X_pool)) / len(X_pool)
        q_star = p_x * np.sqrt(np.abs(uncertainty - R))

        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        q_star[~mask] = 0
        
        if np.sum(q_star) > 0:
            q_star = q_star / np.sum(q_star)
        else:
            q_star[mask] = 1 / np.sum(mask)
        return q_star

    def select_test_samples(self, X_pool, q_proposal, n_samples=1, excluded_indices=None):
        # 提案分布に基づいてテストサンプル選択
        if excluded_indices is None:
            excluded_indices = []
        mask = np.ones(len(X_pool), dtype=bool)
        mask[excluded_indices] = False
        # サンプルの選びやすさ
        q_masked = q_proposal.copy()
        q_masked[~mask] = 0
        if np.sum(q_masked) > 0:
            q_masked = q_masked / np.sum(q_masked)
        else:
            return [],[]
        # index選択
        selected_indices = np.random.choice(
            np.arange(len(X_pool)),
            size=min(n_samples, np.sum(mask)),
            replace=False,
            p=q_masked,
        )
        # 重要度サンプリング
        p_x = np.ones(len(X_pool)) / len(X_pool)
        weights = p_x[selected_indices] / q_masked[selected_indices]

        self.test_indices.extend(selected_indices)
        self.test_weights.extend(weights)
        self.test_proposals.append(q_masked)

        return selected_indices, weights

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
    
    def compute_multi_source_risk(self, X_train, y_train, X_test, y_test, X_pool, test_weights=None):
        # 学習データ、未ラベルデータ、テストデータの情報から評価
        train_risk = cross_entropy_loss(self.model.predict_proba(X_train), y_train)

        pool_probs = self.model.predict_proba(X_pool)
        pool_risk = np.mean(entropy(pool_probs))
        if len(X_test) > 0:
            test_risk = self.estimate_risk(X_test, y_test, weights=test_weights)
        else:
            test_risk = 0

        # 式(5)
        multi_source_risk = (len(X_train) * train_risk + len(X_pool) * pool_risk + len(X_test) * test_risk) / \
                           (len(X_train) + len(X_pool) + len(X_test))
        return multi_source_risk
    
    def integrated_risk_estimation(self, X_quizzes, y_quizzes, quiz_weights, model=None):
        # クイズで得たリスク推定値を信頼度で重みづけ
        if model is None:
            model = self.model
        # 各クイズの信頼度
        confidences = []
        risk_estimates = []
        for X_quiz, y_quiz, weights in zip(X_quizzes, y_quizzes, quiz_weights):
            if len(X_quiz) == 0:
                continue
            risk = self.estimate_risk(X_quiz, y_quiz, weights) # クイズの平均損失
            risk_estimates.append(risk)
            probs = model.predict_proba(X_quiz)
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                # 多クラス分類
                n_classes = probs.shape[1]
                y_one_hot = np.zeros((len(y_quiz), n_classes))
                y_one_hot[np.arange(len(y_quiz)), y_quiz.astype(int)] = 1
                losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1) # 各サンプルの損失
            else:
                losses = -y_quiz * np.log(probs + 1e-10) - (1 - y_quiz) * np.log(1 - probs + 1e-10)
            sq_diff = (losses - risk) ** 2 # 分散計算
            weighted_sq_diff = np.sum(weights * sq_diff) / np.sum(weights) # 重み付き平均
            confidence = 1 / (weighted_sq_diff + 1e-10) # 分散が高いほど信頼度高い
            confidences.append(confidence)
        if not confidences:
            return 0
        confidences = np.array(confidences)
        v_weights = confidences / np.sum(confidences)
        integrated_risk = np.sum(v_weights * np.array(risk_estimates))

        return integrated_risk            
    
    def select_feedback_samples(self, X_test, y_test, X_train, test_indices, q_proposal, n_feedback=1):
        # クイズからフィードバックに使用するサンプル選択(式(9))
        if len(X_test) == 0:
            return []
        probs = self.model.predict_proba(X_test)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # 多クラス分類
            n_classes = probs.shape[1]
            y_one_hot = np.zeros((len(y_test), n_classes))
            y_one_hot[np.arange(len(y_test)), y_test.astype(int)] = 1
            losses = -np.sum(y_one_hot * np.log(probs + 1e-10), axis=1)
        else:
            losses = -y_test * np.log(probs + 1e-10) - (1 - y_test) * np.log(1 - probs + 1e-10)
        
        if len(X_train) > 0:
            dist_matrix = cdist(X_test, X_train) #距離
            min_distances = np.min(dist_matrix, axis=1) # 各テストサンプルがどの訓練データと近いか
            
            # 正規化
            if np.max(min_distances) > 0:
                diversity = min_distances / np.max(min_distances)
            else:
                diversity = np.ones_like(min_distances)
        else:
            diversity = np.ones(len(X_test))

        # 式(9)
        eta = 0.5 # scaling parameter
        feedback_scores = q_proposal[test_indices] * losses + eta * diversity
        selected_indices = np.argsort(feedback_scores)[-n_feedback:][::-1]
        original_indices = test_indices[selected_indices]
        self.feedback_indices.extend(original_indices)
        return original_indices


