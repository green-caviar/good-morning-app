import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import itertools
from collections import deque
import random
from api.services.change_seats.qnetwork import QNetwork
from api.services.change_seats.replay_memory import ReplayMemory

class Agent:

    def __init__(self, state_size, action_size, memory_capacity, learning_rate=1e-3, gamma=0.99):

        self.num_students = state_size # 30
        actual_input_size = self.num_students # 30

        self.action_size = action_size # 435

        self.gamma = gamma

        # 生徒ID(0-29)から2つを選ぶ組み合わせリスト
        self.action_pairs = list(itertools.combinations(range(self.num_students), 2))

        # Agentの脳を作成
        self.model = QNetwork(state_size=actual_input_size, action_size=self.action_size)

        # 学習エンジンの設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # リプレイメモリの設定
        self.memory = ReplayMemory(capacity=memory_capacity)


    def add_experience(self, state, action, reward, next_state, done):
        """エージェントのメモリに経験を追加する"""

        self.memory.push(state, action, reward, next_state, done)


    def learn(self, batch_size):
        """メモリからランダムサンプリングした経験バッチを使って学習する()"""

        if len(self.memory) < batch_size:
            return

        experiences = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # states, next_states は (batch_size, 30) のTensorになる
        states_tensor = torch.from_numpy(np.vstack(states)).float()
        next_states_tensor = torch.from_numpy(np.vstack(next_states)).float()
        actions_tensor = torch.from_numpy(np.vstack(actions)).long()
        rewards_tensor = torch.from_numpy(np.vstack(rewards)).float()
        dones_tensor = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # 正解の計算
        Q_targets_next = self.model(next_states_tensor).max(1)[0].unsqueeze(1)
        Q_targets = rewards_tensor + (self.gamma * Q_targets_next * (1 - dones_tensor))

        # 予測の計算
        Q_predicted = self.model(states_tensor).gather(1, actions_tensor)

        # 誤差の計算とモデル更新
        loss = F.mse_loss(Q_predicted, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def act(self, state, temperature=0.1):
        """ソフトマックス選択の実装"""

        # (30,) のNumPy配列を (30,) のTensorに変換、.unsqueeze(0) で (1, 30) のバッチ形式に変換
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        self.model.eval()
        with torch.no_grad():

            # Q値を予測
            q_values = self.model(state_tensor)

        # ソフトマックス関数でQ値を確率に変換
        probs = F.softmax(q_values[0] / temperature, dim=0)

        # 確率に基づいて行動インデックスを決定
        action_index = torch.multinomial(probs, num_samples=1).item()

        self.model.train()

        return action_index