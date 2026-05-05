import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque

# ==========================================
# 1. 基础 GaussianHMM (复用之前的逻辑，稍作修改支持热启动)
# ==========================================
class GaussianHMM_Scratch:
    def __init__(self, n_components=4, n_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # 模型参数
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        self.is_fitted = False  # 标记是否已经初始化过

    def _init_params(self, X):
        """
        初始化参数。
        如果 is_fitted 为 True，说明是热启动，保留现有参数，不重新随机初始化。
        """
        if self.is_fitted:
            return

        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        self.startprob_ = np.random.rand(self.n_components)
        self.startprob_ /= self.startprob_.sum()

        self.transmat_ = np.random.rand(self.n_components, self.n_components)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # 简单的 KMeans 初始化思想
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        self.covars_ = np.ones((self.n_components, n_features))

        self.is_fitted = True

    def _gaussian_pdf(self, X, mean, var):
        eps = 1e-9
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
        exponent = -0.5 * ((X - mean) ** 2) / (var + eps)
        return np.prod(coeff * np.exp(exponent), axis=1)

    def _compute_emission_probs(self, X):
        n_samples = X.shape[0]
        B = np.zeros((n_samples, self.n_components))
        for i in range(self.n_components):
            B[:, i] = self._gaussian_pdf(X, self.means_[i], self.covars_[i])
        return B + 1e-9

    def fit(self, X):
        """训练模型"""
        # 如果是第一次，必须有足够的数据来初始化
        if not self.is_fitted and X.shape[0] < self.n_components:
            raise ValueError("Not enough data to initialize model")

        self._init_params(X)
        n_samples = X.shape[0]
        last_log_likelihood = -np.inf

        for it in range(self.n_iter):
            # --- E-Step ---
            B = self._compute_emission_probs(X)

            # Forward
            alpha = np.zeros((n_samples, self.n_components))
            scales = np.zeros(n_samples)

            alpha[0] = self.startprob_ * B[0]
            scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-9)
            alpha[0] *= scales[0]

            for t in range(1, n_samples):
                alpha[t] = np.dot(alpha[t - 1], self.transmat_) * B[t]
                scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-9)
                alpha[t] *= scales[t]

            log_likelihood = -np.sum(np.log(scales + 1e-9))

            # Backward
            beta = np.zeros((n_samples, self.n_components))
            beta[-1] = 1.0 * scales[-1]

            for t in range(n_samples - 2, -1, -1):
                beta[t] = np.dot(self.transmat_, (B[t + 1] * beta[t + 1]))
                beta[t] *= scales[t]

            # Gamma & Xi
            gamma = alpha * beta
            gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-9)

            xi = np.zeros((n_samples - 1, self.n_components, self.n_components))
            for t in range(n_samples - 1):
                num = alpha[t].reshape(-1, 1) * self.transmat_ * B[t + 1].reshape(1, -1) * beta[t + 1].reshape(1, -1)
                xi[t] = num / (np.sum(num) + 1e-9)

            # --- M-Step ---
            self.startprob_ = gamma[0]

            sum_xi = np.sum(xi, axis=0)
            sum_gamma_trunc = np.sum(gamma[:-1], axis=0).reshape(-1, 1)
            self.transmat_ = sum_xi / (sum_gamma_trunc + 1e-9)

            sum_gamma = np.sum(gamma, axis=0)
            for i in range(self.n_components):
                weight = gamma[:, i].reshape(-1, 1)
                self.means_[i] = np.sum(X * weight, axis=0) / (sum_gamma[i] + 1e-9)
                diff = X - self.means_[i]
                self.covars_[i] = np.sum((diff ** 2) * weight, axis=0) / (sum_gamma[i] + 1e-9)

            if abs(log_likelihood - last_log_likelihood) < self.tol:
                break
            last_log_likelihood = log_likelihood

        return self

    def predict(self, X):
        """Viterbi 解码"""
        n_samples = X.shape[0]
        B = self._compute_emission_probs(X)
        log_start = np.log(self.startprob_ + 1e-9)
        log_trans = np.log(self.transmat_ + 1e-9)
        log_B = np.log(B + 1e-9)

        delta = np.zeros((n_samples, self.n_components))
        psi = np.zeros((n_samples, self.n_components), dtype=int)

        delta[0] = log_start + log_B[0]

        for t in range(1, n_samples):
            for j in range(self.n_components):
                temp = delta[t - 1] + log_trans[:, j]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + log_B[t, j]

        path = np.zeros(n_samples, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(n_samples - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


# ==========================================
# 2. 在线监测器 (Online Monitor)
# ==========================================
class OnlineHMM_Monitor:
    def __init__(self, window_size=1, n_components=4):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)  # 自动丢弃旧数据的队列
        self.hmm = GaussianHMM_Scratch(n_components=n_components, n_iter=20)  # 迭代次数可以少一点，因为是热启动
        self.scaler = StandardScaler()
        self.history_raw = []  # 仅用于画图记录
        self.detected_states = []  # 记录每一步检测到的状态

    def update(self, new_data_point):
        """
        接收一个新的联邦学习 Round 数据
        new_data_point: [loss, grad, cons]
        """
        # 1. 记录原始数据
        self.history_raw.append(new_data_point)

        # 2. 加入滑动窗口
        self.buffer.append(new_data_point)

        # 3. 如果数据不够填满窗口，先不训练，或者只做简单处理
        if len(self.buffer) < self.window_size:
            self.detected_states.append(-1)  # -1 表示数据不足，未定义
            print("OnlineHMM====>表示数据不足，未定义")
            return -1

        print(f"OnlineHMM====>准备训练数据")
        # 4. 准备训练数据
        X_window = np.array(np.vstack(self.buffer))

        # 注意：在线场景下，Scaler 需要谨慎处理。
        # 这里为了演示简单，每次对窗口内数据重新标准化。
        # 实际生产中应该用 IncrementalScaler (partial_fit)。
        X_scaled = self.scaler.fit_transform(X_window)

        # 5. 训练 HMM (热启动：利用上一次的参数继续优化)
        self.hmm.fit(X_scaled)

        # 6. 预测当前最新点的状态
        # 我们对整个窗口预测，但只取最后一个点（即最新点）的状态
        hidden_states = self.hmm.predict(X_scaled)
        current_state = hidden_states[-1]

        self.detected_states.append(current_state)
        return current_state


# ==========================================
# 3. 模拟联邦学习流式数据输入
# ==========================================

if __name__ == '__main__':
    # 生成模拟数据 (和之前一样)
    np.random.seed(42)


    def generate_phase_data(length, loss_mean, grad_mean, cons_mean, noise_level):
        loss = np.random.normal(loss_mean, noise_level * 0.1, length)
        grad = np.random.normal(grad_mean, noise_level * 2.0, length)
        cons = np.random.normal(cons_mean, noise_level * 0.1, length)
        cons = np.clip(cons, 0, 1)
        grad = np.abs(grad)
        return np.column_stack([loss, grad, cons])


    data_phase_0 = generate_phase_data(50, 0.5, 50.0, 0.2, 1.0)
    data_phase_1 = generate_phase_data(50, 0.2, 20.0, 0.5, 0.8)
    data_phase_2 = generate_phase_data(50, 0.05, 5.0, 0.8, 0.5)
    data_phase_3 = generate_phase_data(50, 0.001, 1.0, 0.95, 0.1)
    X_stream = np.vstack([data_phase_0, data_phase_1, data_phase_2, data_phase_3])

    # ==========================================
    # 4. 运行在线监测
    # ==========================================
    print("开始联邦学习在线监测...")
    monitor = OnlineHMM_Monitor(window_size=30, n_components=4)

    for t, data_point in enumerate(X_stream):
        # 模拟：数据一个接一个地来
        state = monitor.update(data_point)

        if t % 20 == 0:
            print(f"Round {t}: Data={np.round(data_point, 2)} -> Detected State: {state}")

    print("监测结束。")

    # ==========================================
    # 5. 可视化在线监测结果
    # ==========================================
    history = np.array(monitor.history_raw)
    states = np.array(monitor.detected_states)

    plt.figure(figsize=(12, 8))
    titles = ["Loss", "Gradient Norm", "Consistency"]
    colors = ['blue', 'green', 'orange']

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(history[:, i], color=colors[i], label=titles[i], linewidth=1.5)

        # 绘制背景颜色 (跳过前面数据不足的阶段)
        valid_indices = np.where(states != -1)[0]
        if len(valid_indices) > 0:
            for t in range(valid_indices[0], len(states) - 1):
                plt.axvspan(t, t + 1, facecolor=plt.cm.Pastel1(states[t]), alpha=0.3, edgecolor=None)

        plt.title(f"Real-time Feature: {titles[i]}")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

    plt.xlabel("FL Rounds (Streaming)")
    plt.tight_layout()
    plt.show()