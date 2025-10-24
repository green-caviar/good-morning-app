import torch
import numpy as np
import itertools
from collections import deque

from api.services.change_seats.env import Env
from api.services.change_seats.agent import Agent


# 学習の基本設定
NUM_EPISODES = 1000
MAX_STEPS = 100
BATCH_SIZE = 32
GAMMA = 0.99

# ソフトマックス選択の温度設定
TEMP_START = 5.0
TEMP_END = 1.0
TEMP_DECAY = 0.995 #はじめ温度を高くし、徐々に低くすることで効率的な学習を促進

print("環境とエージェントを初期化します...")

try:
    env = Env()
except FileNotFoundError:
    print("エラー: 'relationships.csv' が見つかりません。train.py と同じ階層に配置してください。")
    exit()

# 定数の設定
NUM_STUDENTS = 30
ACTION_SIZE = len(list(itertools.combinations(range(NUM_STUDENTS), 2))) # 435

agent = Agent(state_size=NUM_STUDENTS,
              action_size=ACTION_SIZE,
              memory_capacity=50000,
              learning_rate=1e-3,
              gamma=GAMMA)

# デバッグ用プリント
print(f"Agent初期化完了。脳への入力サイズ: {agent.model.layer1.in_features} (30のはず)")


# メインの学習ループ
temperature = TEMP_START
scores_deque = deque(maxlen=100)
all_scores = []

print("学習を開始します...")
for episode in range(1, NUM_EPISODES + 1):


    # 環境をリセット
    state_array = env.reset()

    initial_score = env.calculate_score(state_array)
    episode_score_change = 0 # 報酬(スコアの変化量)の合計を記録

    for step in range(MAX_STEPS):

        # Agentに渡すためにフラット化
        state_1d = state_array.flatten()

        action_index = agent.act(state_1d, temperature)

        """経験"""
        action_pair = agent.action_pairs[action_index]

        next_state_array, reward, done = env.step(state_array, action_pair)

        """記憶"""
        # Agentのメモリに保存するためにフラット化
        next_state_1d = next_state_array.flatten()

        # メモリには1Dの座席表を記憶させる
        agent.add_experience(state_1d, action_index, reward, next_state_1d, done)

        """学習"""
        agent.learn(BATCH_SIZE)

        # 状態(座席表)を更新
        state_array = next_state_array
        episode_score_change += reward

        if done: # (今回は使わない)
            break

    #エピソード終了
    final_score = env.calculate_score(state_array)

    all_scores.append(final_score)
    scores_deque.append(final_score)

    # 温度を下げる
    temperature = max(TEMP_END, TEMP_DECAY * temperature)

    # 100エピソードごとに進捗を表示
    if episode % 100 == 0:
        avg_score = np.mean(scores_deque)
        print(f"\nエピソード {episode}/{NUM_EPISODES} | 直近100回の平均スコア: {avg_score:.2f}")
    else:
        # 進捗がわかるようにドット(.)を表示
        print(f".", end="")

print("\n学習が完了しました。")

# 学習が完了したAgentの脳を保存
torch.save(agent.model.state_dict(), 'qnetwork.pth')
print("学習済みモデルを 'qnetwork.pth' として保存しました。")