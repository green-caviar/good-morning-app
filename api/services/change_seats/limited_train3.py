# api/services/change_seats/limited_train.py (★30入力テスト版★)

import torch
import numpy as np
import itertools
from collections import deque
import time
import sys
import pandas as pd

try:
    from .env import Env
    from .agent import Agent
    from . import config
    # from . import matrix_utils # (使わない)
except ImportError:
    from env import Env
    from agent import Agent
    import config
    # import matrix_utils # (使わない)

# --- (学習設定は同じ) ---
NUM_EPISODES = 1000
MAX_STEPS = 100
BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_CAPACITY = 10000
# ... (TEMPなども同じ) ...
TEMP_START = 5.0
TEMP_END = 0.1 
TEMP_DECAY = 0.995

print("--- ステップ4-C: Day 5 回帰テスト (30入力) ---")

# --- 0. CSVから「完全な30x30マトリクス」をロード ---
print(f"'{config.CSV_PATH}' から「完全な(30, 30)」マトリクスを読み込みます...")
try:
    df = pd.read_csv(config.CSV_PATH, header=0, index_col=0)
    fixed_matrix = df.values.astype(np.float32) # (30, 30)
    if fixed_matrix.shape != (config.NUM_STUDENTS, config.NUM_STUDENTS):
         raise ValueError("CSVの形状が(30,30)ではありません。")
    print(f"固定ルール (Shape: {fixed_matrix.shape}) の読み込み完了。")
except Exception as e:
    print(f"致命的エラー: CSVの読み込みに失敗しました。 {e}")
    sys.exit(1)

# --- 1. ★★★ Agentの初期化 (30入力) ★★★ ---
state_size = config.NUM_STUDENTS # 30
action_size = len(list(itertools.combinations(range(config.NUM_STUDENTS), 2))) # 435

agent = Agent(
    state_size=state_size, # (seating_size と relations_size をやめる)
    action_size=action_size,
    memory_capacity=10000,
    learning_rate=1e-3,
    gamma=0.99
)
# agent.actual_input_size の代わりに agent.model.layer1.in_features で確認
print(f"Agent初期化完了 (入力: {agent.model.layer1.in_features} / 行動: {agent.action_size})")


# --- 2. メインの学習ループ ---
temperature = TEMP_START
scores_deque = deque(maxlen=100)
all_scores = []
start_time = time.time()

# 「ルール固定」のEnvを1回だけ作成
env = Env(relations_matrix=fixed_matrix)
print("学習ループを開始します (ルール固定 / 30入力)...")

for episode in range(1, NUM_EPISODES + 1):
    
    seating_array_2d = env.reset() # (6, 5)

    for step in range(MAX_STEPS):

        # --- 3. ★★★ 30要素の「状態」を構築 ★★★ ---
        seating_array_1d = seating_array_2d.flatten() # 30要素
        # full_state = np.concatenate(...) # (900要素の連結をやめる)
        
        # --- 4. Agentに行動を選択させる (30要素を渡す) ---
        action_index = agent.act(seating_array_1d, temperature) # (full_state をやめる)
        action_pair = agent.action_pairs[action_index]

        # --- 5. 環境を1ステップ進める ---
        next_seating_array_2d, reward, done = env.step(seating_array_2d, action_pair)

        # --- 6. ★★★ 30要素の「次の状態」を構築 ★★★ ---
        next_seating_array_1d = next_seating_array_2d.flatten()
        # next_full_state = np.concatenate(...) # (900要素の連結をやめる)

        # --- 7. メモリに経験を追加 (30要素のstateを記憶) ---
        agent.add_experience(
            seating_array_1d,    # (full_state をやめる)
            action_index, 
            reward, 
            next_seating_array_1d, # (next_full_state をやめる)
            done
        )

        # --- 8. 学習 ---
        agent.learn(BATCH_SIZE)

        seating_array_2d = next_seating_array_2d
        if done:
            break
            
    # --- (エピソード終了後の処理は変更なし) ---
    final_score = env.calculate_score(seating_array_2d)
    scores_deque.append(final_score)
    all_scores.append(final_score)
    temperature = max(TEMP_END, TEMP_DECAY * temperature)
    if episode % 10 == 0:
        avg_score = np.mean(scores_deque)
        elapsed = time.time() - start_time
        print(f"Epi: {episode}/{NUM_EPISODES} | AvgScore(100): {avg_score:.2f} | Temp: {temperature:.2f} | Time: {elapsed:.1f}s")

# ... (モデルの保存処理) ...
print("\n--- 30入力テスト学習が完了しました ---")
MODEL_PATH = "qnetwork_TEST_30_INPUT.pth"
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"テストモデルを '{MODEL_PATH}' として保存しました。")