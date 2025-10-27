# api/services/change_seats/limited_train.py (★テスト版★)

import torch
import numpy as np
import itertools
from collections import deque
import time
import sys
import pandas as pd # <--- ★Pandasを追加

# --- モジュールをインポート ---
try:
    from .env import Env
    from .agent import Agent
    from . import config
    # from . import matrix_utils # <--- ★使わない
except ImportError:
    from env import Env
    from agent import Agent
    import config
    # import matrix_utils # <--- ★使わない

# --- (学習設定は同じ) ---
NUM_EPISODES = 1000
MAX_STEPS = 100
BATCH_SIZE = 32
# ... (TEMPなども同じ) ...
TEMP_START = 5.0
TEMP_END = 0.1 
TEMP_DECAY = 0.995

print("--- ステップ4-B: 問題単純化テスト (CSV固定ルール) ---")

# --- 0. ★★★ CSVから「完全な30x30マトリクス」をロード ★★★ ---
print(f"'{config.CSV_PATH}' から「完全な(30, 30)」マトリクスを読み込みます...")
try:
    df = pd.read_csv(config.CSV_PATH, header=0, index_col=0)
    fixed_matrix = df.values.astype(np.float32) # (30, 30)
    
    if fixed_matrix.shape != (config.NUM_STUDENTS, config.NUM_STUDENTS):
         raise ValueError(f"CSVの形状が(30,30)ではありません。")
         
    fixed_matrix_flat = fixed_matrix.flatten() # 900要素
    print(f"固定ルール (Shape: {fixed_matrix.shape}) の読み込み完了。")
except Exception as e:
    print(f"致命的エラー: CSVの読み込みに失敗しました。 {e}")
    sys.exit(1)

# --- 1. Agentの初期化 (930入力対応版) ---
# ... (変更なし。Agentの初期化は同じ) ...
seating_size = config.NUM_STUDENTS
relations_size = config.NUM_STUDENTS * config.NUM_STUDENTS
action_size = len(list(itertools.combinations(range(config.NUM_STUDENTS), 2))) 
agent = Agent(
    seating_size=seating_size,
    relations_size=relations_size,
    action_size=action_size,
    memory_capacity=10000,
    learning_rate=1e-3,
    gamma=0.99
)
print(f"Agent初期化完了 (入力: {agent.actual_input_size} / 行動: {agent.action_size})")


# --- 2. メインの学習ループ ---
temperature = TEMP_START
scores_deque = deque(maxlen=100)
all_scores = []
start_time = time.time()

# ★★★ 「ルール固定」のEnvを1回だけ作成 ★★★
env = Env(relations_matrix=fixed_matrix)
print("学習ループを開始します (ルール完全固定)...")

for episode in range(1, NUM_EPISODES + 1):

    # --- (a) (b) (c) は削除 (Envはループの外で作成済み) ---
    
    # 環境をリセット (6, 5) の座席表
    seating_array_2d = env.reset()

    for step in range(MAX_STEPS):

        # --- 3. 930要素の「完全な状態」を構築 ---
        seating_array_1d = seating_array_2d.flatten() # 30要素
        
        # [座席表(30)] + [固定マトリクス(900)] を連結
        full_state = np.concatenate((seating_array_1d, fixed_matrix_flat))
        
        # --- 4. Agentに行動を選択させる ---
        action_index = agent.act(full_state, temperature)
        action_pair = agent.action_pairs[action_index]

        # --- 5. 環境を1ステップ進める ---
        next_seating_array_2d, reward, done = env.step(seating_array_2d, action_pair)

        # --- 6. 930要素の「次の完全な状態」を構築 ---
        next_seating_array_1d = next_seating_array_2d.flatten()
        next_full_state = np.concatenate((next_seating_array_1d, fixed_matrix_flat))

        # --- 7. メモリに経験を追加 ---
        agent.add_experience(full_state, action_index, reward, next_full_state, done)

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

# ... (モデルの保存処理は同じ) ...
print("\n--- テスト学習が完了しました ---")
MODEL_PATH = "qnetwork_TEST_FIXED.pth"
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"テストモデルを '{MODEL_PATH}' として保存しました。")