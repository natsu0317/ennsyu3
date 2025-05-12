# ALとATの統合
import numpy as np
import copy
from active_learning.uncertainty_learner import UncertaintyActiveLearner
from active_testing.active_tester import ActiveTester
from utils.losses import cross_entropy_loss, entropy

class ATLFramework:
    def __init__(self, model, X_pool, y_pool=None, initial_labeled_indices=None, 
             test_frequency=1, test_batch_size=10, feedback_ratio=0.5, window_size=3):
        self.model = model
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.active_learner = UncertaintyActiveLearner(model)
        self.active_tester = ActiveTester(model)
        
        # ラベル付きインデックスを初期化
        if initial_labeled_indices is None:
            self.labeled_indices = []
        else:
            self.labeled_indices = list(initial_labeled_indices)
            self.active_learner.labeled_indices = list(initial_labeled_indices)
        
        # テストとフィードバックのインデックスを初期化
        self.test_indices = []
        self.feedback_indices = []
        
        # パラメータ
        self.test_frequency = test_frequency  # テストを実行する頻度
        self.test_batch_size = test_batch_size  # クイズごとのテストサンプル数
        
        # データセットサイズに基づいてフィードバック比率を調整
        if len(X_pool) > 10000:
            # 大規模データセットでは、より積極的なフィードバック
            self.feedback_ratio = min(0.7, feedback_ratio * 1.2)
        else:
            self.feedback_ratio = feedback_ratio
            
        # ウィンドウサイズを調整
        self.window_size = max(3, window_size)  # 最低3ラウンドの履歴を確保
        
        # 履歴
        self.risk_history = []
        self.integrated_risk_history = []
        self.true_risk_history = []  # 評価用のみ
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
        """
        アクティブクイズを実行してモデルを評価します
        
        Parameters:
        -----------
        al_round : int
            現在のアクティブラーニングラウンド
        
        Returns:
        --------
        selected_indices : list
            選択されたテストサンプルのインデックスリスト
        """
        # 未ラベルデータを取得
        X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
        
        if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
            return []
        
        # 現在のラベル付きデータとテストデータを取得
        X_labeled, y_labeled = self.get_labeled_data()
        X_test, y_test = self.get_test_data()
        
        # 多ソースリスク推定を計算
        # 論文の式(5)に忠実に実装
        train_weight = len(X_labeled)
        pool_weight = len(X_unlabeled)
        test_weight = len(X_test)
        
        # 訓練リスク（より正確に計算）
        if len(X_labeled) > 0:
            # K分割交差検証でより正確な訓練リスクを推定
            n_splits = min(5, len(X_labeled))
            train_risks = []
            
            # データをシャッフル
            indices = np.random.permutation(len(X_labeled))
            X_labeled_shuffled = X_labeled[indices]
            y_labeled_shuffled = y_labeled[indices]
            
            # 簡易的なK分割交差検証
            split_size = len(X_labeled) // n_splits
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X_labeled)
                
                # 検証セットとトレーニングセットを分割
                val_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                # 一時的なモデルを訓練
                temp_model = copy.deepcopy(self.model)
                temp_model.fit(X_labeled[train_indices], y_labeled[train_indices])
                
                # 検証セットでリスクを計算
                val_probs = temp_model.predict_proba(X_labeled[val_indices])
                val_risk = cross_entropy_loss(val_probs, y_labeled[val_indices])
                train_risks.append(val_risk)
            
            train_risk = np.mean(train_risks)
        else:
            train_risk = 0
        
        # プールリスク
        pool_probs = self.model.predict_proba(X_unlabeled)
        pool_risk = np.mean(entropy(pool_probs))
        
        # テストリスク
        if len(X_test) > 0:
            test_weights = np.ones(len(X_test)) / len(X_test)  # 均等な重み
            test_risk = self.active_tester.estimate_risk(X_test, y_test, test_weights)
        else:
            test_risk = 0
        
        # 多ソースリスク推定を計算
        total_weight = train_weight + pool_weight + test_weight
        if total_weight > 0:
            multi_source_risk = (train_weight * train_risk + pool_weight * pool_risk + test_weight * test_risk) / total_weight
        else:
            multi_source_risk = 0
        
        # テスト提案を計算
        excluded_indices = self.labeled_indices + self.test_indices
        q_proposal = self.active_tester.compute_test_proposal(
            self.X_pool, excluded_indices, multi_source_risk
        )
        
        # テストサンプルを選択
        selected_indices, weights = self.active_tester.select_test_samples(
            self.X_pool, q_proposal, n_samples=self.test_batch_size, excluded_indices=excluded_indices
        )
        
        # テストインデックスを更新
        self.test_indices.extend(selected_indices)
        
        # 選択されたテストデータを取得
        X_quiz = self.X_pool[selected_indices]
        y_quiz = self.y_pool[selected_indices] if self.y_pool is not None else None
        
        # クイズデータを後の統合のために保存
        self.quiz_X.append(X_quiz)
        self.quiz_y.append(y_quiz)
        self.quiz_weights.append(weights)
        
        # このクイズのリスクを推定
        quiz_risk = self.active_tester.estimate_risk(X_quiz, y_quiz, weights)
        self.risk_history.append(quiz_risk)

        # フィードバックサンプルを選択
        n_feedback = int(self.feedback_ratio * len(selected_indices))
        if n_feedback > 0:
            feedback_indices = self.active_tester.select_feedback_samples(
                X_quiz, y_quiz, X_labeled, selected_indices, q_proposal, n_feedback=n_feedback
            )
            
            # フィードバックインデックスを更新
            self.feedback_indices.extend(feedback_indices)
            
            # フィードバックサンプルをラベル付きセットに追加
            self.labeled_indices.extend(feedback_indices)
            self.active_learner.labeled_indices.extend(feedback_indices)
            
            # フィードバックサンプルをテストインデックスから削除
            self.test_indices = [idx for idx in self.test_indices if idx not in feedback_indices]
            
            # フィードバック後に残ったテストサンプルでリスク推定を再計算
            # これにより、フィードバック後のテストセット分布でのリスク推定が得られる
            remaining_indices = [i for i, idx in enumerate(selected_indices) if idx not in feedback_indices]
            if remaining_indices:
                X_remaining = X_quiz[remaining_indices]
                y_remaining = y_quiz[remaining_indices]
                weights_remaining = weights[remaining_indices]
                # 重みを再正規化
                weights_remaining = weights_remaining / np.sum(weights_remaining)
                # リスクを再推定
                quiz_risk = self.active_tester.estimate_risk(X_remaining, y_remaining, weights_remaining)
                # 履歴を更新
                self.risk_history[-1] = quiz_risk

        # 統合リスク推定値を計算
        integrated_risk = self.active_tester.integrated_risk_estimation(
            self.quiz_X, self.quiz_y, self.quiz_weights
        )
        self.integrated_risk_history.append(integrated_risk)
        
        # 真のリスクを計算（評価用）
        if self.y_pool is not None:
            # 全データに対する真のリスクを計算
            true_risk = cross_entropy_loss(self.model.predict_proba(self.X_pool), self.y_pool)
            self.true_risk_history.append(true_risk)
        
        return selected_indices
    
    def check_early_stopping(self):
        """
        早期停止条件を満たしているかチェックします（付録Aの表記法に基づく）
        
        Returns:
        --------
        stop : bool
            早期停止条件を満たしている場合はTrue
        """
        if len(self.integrated_risk_history) < self.window_size + 1:
            return False
            
        # 統合リスクの移動平均の変化を計算
        window = self.window_size
        current_avg = np.mean(self.integrated_risk_history[-window:])
        previous_avg = np.mean(self.integrated_risk_history[-(window+1):-1])
        delta_risk = abs(current_avg - previous_avg)
        
        # 未ラベルデータを取得
        X_unlabeled, _ = self.get_unlabeled_data()
        
        # 安定化予測（SP）を計算
        if len(X_unlabeled) > 0:
            # 現在のモデル予測を取得
            current_probs = self.model.predict_proba(X_unlabeled)
            
            # 最新のサンプルなしで一時的なモデルを訓練
            temp_model = copy.deepcopy(self.model)
            X_prev_labeled = self.X_pool[self.labeled_indices[:-self.test_batch_size]]
            y_prev_labeled = self.y_pool[self.labeled_indices[:-self.test_batch_size]]
            temp_model.fit(X_prev_labeled, y_prev_labeled)
            
            # 以前の予測を取得
            prev_probs = temp_model.predict_proba(X_unlabeled)
            
            # 予測変化を計算
            pred_changes = np.mean(np.abs(current_probs - prev_probs))
            sp = 1 - pred_changes
        else:
            sp = 1.0
            
        # 付録Aのλパラメータを使用して組み合わせた停止基準
        lambda_param = 0.5  # リスク推定と未ラベル情報のバランスを取るパラメータ
        combined_criterion = lambda_param * delta_risk + (1 - lambda_param) * (1 - sp)
        
        threshold = 0.01  # 組み合わせた基準のしきい値
        
        return combined_criterion < threshold

    def run_active_learning(self, n_rounds=10, n_samples_per_round=10):
        """
        アクティブラーニングプロセスを実行します
        
        Parameters:
        -----------
        n_rounds : int, optional
            アクティブラーニングのラウンド数
        n_samples_per_round : int, optional
            ラウンドごとに選択するサンプル数
        
        Returns:
        --------
        model : BaseModel
            学習済みモデル
        """
        for al_round in range(n_rounds):
            # print(f"Active Learning Round {al_round+1}/{n_rounds}")
            
            # 現在のラベル付きデータを取得
            X_labeled, y_labeled = self.get_labeled_data()
            
            # ラベル付きデータでモデルを学習
            if len(X_labeled) > 0 and y_labeled is not None:
                # 学習の初期段階ではエポック数を増やす
                if al_round < 5 and hasattr(self.model, 'epochs'):
                    original_epochs = self.model.epochs
                    self.model.epochs = original_epochs * 2
                    self.model.fit(X_labeled, y_labeled)
                    self.model.epochs = original_epochs
                else:
                    self.model.fit(X_labeled, y_labeled)
            
            # テスト頻度に応じてアクティブテストを実行
            if al_round % self.test_frequency == 0:
                # print(f"  Performing active quiz...")
                self.perform_active_quiz(al_round)
                
                # 早期停止をチェック
                if self.check_early_stopping():
                    # print(f"  Early stopping criteria met at round {al_round+1}")
                    break
            
            # アクティブラーニングのサンプル選択
            X_unlabeled, unlabeled_indices = self.get_unlabeled_data()
            if len(X_unlabeled) == 0 or len(unlabeled_indices) == 0:
                # print("  No more unlabeled data available.")
                break
                
            selected_indices = self.active_learner.select_samples(
                X_unlabeled, n_samples=n_samples_per_round, excluded_indices=[]
            )
            
            # 選択されたインデックスを元のプールのインデックスに変換
            original_indices = [unlabeled_indices[i] for i in selected_indices if i < len(unlabeled_indices)]
            
            # 選択されたサンプルがない場合の処理
            if not original_indices:
                # print("  No more informative samples available.")
                break
                
            # ラベル付きインデックスを更新
            self.labeled_indices.extend(original_indices)
            self.active_learner.labeled_indices.extend(original_indices)
            
            # 現在の統計を表示
            # if self.y_pool is not None and len(self.integrated_risk_history) > 0:
                # print(f"  Labeled samples: {len(self.labeled_indices)}")
                # print(f"  Test samples: {len(self.test_indices)}")
                # print(f"  Feedback samples: {len(self.feedback_indices)}")
                # print(f"  Estimated risk: {self.integrated_risk_history[-1]:.4f}")
                # if len(self.true_risk_history) > 0:
                    # print(f"  True risk: {self.true_risk_history[-1]:.4f}")
                    # print(f"  Estimation error: {abs(self.integrated_risk_history[-1] - self.true_risk_history[-1]):.4f}")
        
        # 最終的なモデルを訓練（すべてのラベル付きデータを使用）
        X_labeled, y_labeled = self.get_labeled_data()
        if len(X_labeled) > 0 and y_labeled is not None:
            # 最終学習ではエポック数を増やす
            if hasattr(self.model, 'epochs'):
                original_epochs = self.model.epochs
                self.model.epochs = original_epochs * 3
                self.model.fit(X_labeled, y_labeled)
                self.model.epochs = original_epochs
            else:
                self.model.fit(X_labeled, y_labeled)
        
        return self.model