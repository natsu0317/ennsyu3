# ALとATの統合
import numpy as np
import copy
from active_learning.uncertainty_learner import UncertaintyActiveLearner
from active_testing.active_tester import ActiveTester
from utils.losses import cross_entropy_loss

class ATLFramework:
    def __init__(self, model, X_pool, y_pool=None, initial_labeled_indices=None, 
                 test_frequency=1, test_batch_size=10, feedback_ratio=0.5, window_size=3):
        self.model = model
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.active_learner = UncertaintyActiveLearner(model)
        self.active_tester = ActiveTester(model)

        if initial_labeled_indices is None:
            self.labeled_indices = []
        else:
            self.labeled_indices = list(initial_labeled_indices)
            self.active_learner.labeled_indices = list(initial_labeled_indices)

        self.test_indices = []
        self.feedback_indices = []

        self.test_frequency = test_frequency
        self.test_batch_size = test_batch_size # クイズのサンプル数
        self.feedback_ratio = feedback_ratio # feedbackとして使用するtest sampleの割合
        self.window_size = window_size # 早期停止時に何回分の履歴を見るか

        self.risk_history = []
        self.integrated_risk_history = []
        self.true_risk_history = []
        self.quiz_X = []
        self.quiz_y = []
        self.quiz_weights = []

    def get_labeled_data(self):
        # ラベル付きデータ
        X_labeled = self.X_pool[self.labeled_indices]
        y_labeled = self.y_pool[self.labeled_indices] if self.y_pool is not None else None
        return X_labeled, y_labeled
    
    def get_test_data(self):
        # テストデータ取得
        X_test = self.X_pool[self.test_indices]
        y_test = self.y_pool[self.test_indices] if self.y_pool is not None else None
        return X_test, y_test
    
    def get_feedback_data(self):
        X_feedback = self.X_pool[self.feedback_indices]
        y_feedback = self.y_pool[self.feedback_indices] if self.y_pool is not None else None
        return X_feedback, y_feedback

    def get_unlabeled_indices(self):
        # 未ラベルデータのindex
        all_labeled = set(self.labeled_indices + self.test_indices)
        return [i for i in range(len(self.X_pool)) if i not in all_labeled]

    def get_unlabeled_data(self):
        # 未ラベルデータ
        unlabeled_indices = self.get_unlabeled_indices()
        X_unlabeled = self.X_pool[unlabeled_indices]
        return X_unlabeled, unlabeled_indices
    
    def perform_active_quiz(self, al_round):
        X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
        if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
            return []
        X_labeled, y_labeled = self.get_labeled_data()
        X_test, y_test = self.get_test_data()
        multi_source_risk = self.active_tester.compute_multi_source_risk(
            X_labeled, y_labeled, X_test, y_test, X_unlabeled
        )
        excluded_indices = self.labeled_indices + self.test_indices
        # テスト提案
        q_proposal = self.active_tester.compute_test_proposal(self.X_pool, excluded_indices, multi_source_risk)
        # テストサンプル選択
        selected_indices, weights = self.active_tester.select_test_samples(
            self.X_pool, q_proposal, n_samples=self.test_batch_size, excluded_indices=excluded_indices
        )
        self.test_indices.extend(selected_indices)
        X_quiz = self.X_pool[selected_indices]
        y_quiz = self.y_pool[selected_indices]

        self.quiz_X.append(X_quiz)
        self.quiz_y.append(y_quiz)
        self.quiz_weights.append(weights)
        # クイズでモデルがどれだけ間違えたか
        quiz_risk = self.active_tester.estimate_risk(X_quiz, y_quiz, weights)
        self.risk_history.append(quiz_risk)
        # これまでのクイズ結果を信頼度で重みづけして合成（全体のリスク推定））
        integrated_risk = self.active_tester.integrated_risk_estimation(
            self.quiz_X, self.quiz_y, self.quiz_weights
        )
        self.integrated_risk_history.append(integrated_risk)
        # 全データの正解ラベルがある場合：本当のリスク計算
        if self.y_pool is not None:
            true_risk = cross_entropy_loss(self.model.predict_proba(self.X_pool), self.y_pool)
            self.true_risk_history.append(true_risk)
        
        n_feedback = int(self.feedback_ratio * len(selected_indices)) # 何個フィードバックに使用するか
        if n_feedback > 0:
            feedback_indices = self.active_tester.select_feedback_samples(X_quiz, y_quiz, X_labeled, selected_indices, q_proposal, n_feedback=n_feedback)
            self.feedback_indices = feedback_indices
            self.labeled_indices.extend(feedback_indices)
            self.active_learner.labeled_indices.extend(feedback_indices)
            # feedback sampleをtest indexから削除
            self.test_indices = [idx for idx in self.test_indices if idx not in feedback_indices]
        
        return selected_indices
    
    def check_early_stopping(self):
        # 早期停止条件のチェック
        if len(self.integrated_risk_history) < self.window_size + 1:
            return False
        # 移動平均
        window = self.window_size 
        current_avg = np.mean(self.integrated_risk_history[-window:])
        previous_avg = np.mean(self.integrated_risk_history[-(window+1):-1])
        delta_risk = abs(current_avg - previous_avg)

        X_unlabeled, _ = self.get_unlabeled_data()
        # ラベルを追加する前後での予測変化
        if len(X_unlabeled) > 0:
            # 現在のモデル予測
            current_probs = self.model.predict_proba(X_unlabeled)
            # 最後に追加したサンプル以外でモデルをトレーニング
            temp_model = copy.deepcopy(self.model)
            X_prev_labeled = self.X_pool[self.labeled_indices[:-self.test_batch_size]]
            y_prev_labeled = self.y_pool[self.labeled_indices[:-self.test_batch_size]]
            temp_model.fit(X_prev_labeled, y_prev_labeled)

            prev_probs = temp_model.predict_proba(X_unlabeled)
            pred_changes = np.mean(np.abs(current_probs - prev_probs))
            sp = 1 - pred_changes
        else: 
            sp = 1.0
        
        threshold = 0.01
        sp_threshold = 0.95
        return delta_risk < threshold and sp > sp_threshold
    
    def run_active_learning(self, n_rounds=10, n_samples_per_round=10):
        for al_round in range(n_rounds):
            print(f"Active Learning Round {al_round+1}/{n_rounds}")
            X_labeled, y_labeled = self.get_labeled_data()
            
            if len(X_labeled) > 0 and y_labeled is not None:
                self.model.fit(X_labeled, y_labeled)

            if al_round % self.test_frequency == 0:
                print(f"  Performing active quiz...")
                self.perform_active_quiz(al_round)
                
                # 早期停止をチェック
                if self.check_early_stopping():
                    print(f"  Early stopping criteria met at round {al_round+1}")
                    break

            X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
            if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
                print("  No more unlabeled data available.")
                break
            # X_unlabeledにおけるindex
            selected_indices = self.active_learner.select_samples(X_unlabeled, n_samples=n_samples_per_round, excluded_indices=[])  
            # 元のX_poolにおけるindexに変換
            original_indices = [unlabeled_indices[i] for i in selected_indices if i < len(unlabeled_indices)]

            if not original_indices:
                print("  No more informative samples available.")
                break

            self.labeled_indices.extend(original_indices)
            self.active_learner.labeled_indices.extend(original_indices)

             # 現在の統計を表示
            if self.y_pool is not None and len(self.integrated_risk_history) > 0:
                print(f"  Labeled samples: {len(self.labeled_indices)}")
                print(f"  Test samples: {len(self.test_indices)}")
                print(f"  Feedback samples: {len(self.feedback_indices)}")
                print(f"  Estimated risk: {self.integrated_risk_history[-1]:.4f}")
                if len(self.true_risk_history) > 0:
                    print(f"  True risk: {self.true_risk_history[-1]:.4f}")
                    print(f"  Estimation error: {abs(self.integrated_risk_history[-1] - self.true_risk_history[-1]):.4f}")
        
        return self.model

            

