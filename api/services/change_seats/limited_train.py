# api/services/change_seats/limited_train.py (CSV対応版)

import torch
import numpy as np
import itertools
from collections import deque
import time
import sys

# --- ステップ1〜3で作成したモジュールをインポート ---
# (★ここを修正します★)
try:
    # 'limited_train.py' と同じ階層にあるモジュールを
    # 相対パス ( . ) でインポートします
    from .env import Env
    from .agent import Agent
    from . import config         # NG: from .config import config
    from . import matrix_utils   # NG: from .matrix_utils import matrix_utils

except ImportError:
    # (この except ブロックは、 -m で実行する限り使われません)
    print("--- インポートエラー: exceptブロックが実行されました ---")
    from env import Env
    from agent import Agent
    import config
    import matrix_utils

# --- 学習の基本設定 ---
NUM_EPISODES = 1000  # 学習エピソード数 (テスト時は 100 など小さく)
MAX_STEPS = 100      # 1エピソードあたりの最大席替え回数
BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_CAPACITY = 10000 # メモリ容量 (テスト時は 1000 など小さく)

# ソフトマックス選択の温度設定
TEMP_START = 5.0
TEMP_END = 0.1 
TEMP_DECAY = 0.995

print("--- ステップ4: 限定的汎用AI 学習開始 (CSV対応版) ---")

# --- 0. ★★★ CSVから「固定データ」をロード ★★★ ---
print(f"'{config.CSV_PATH}' から固定データを読み込みます...")
try:
    # 学習開始前に1回だけCSVを読み込み、(29, 29) の固定データを取得
    fixed_data = matrix_utils.load_fixed_data_from_csv()
    print(f"固定データ (Shape: {fixed_data.shape}) の読み込み完了。")
except Exception as e:
    print(f"致命的エラー: 固定データの読み込みに失敗しました。 {e}")
    print("CSVファイルのパスと内容を確認してください。")
    sys.exit(1) # スクリプトを終了

# --- 1. Agentの初期化 (930入力対応版) ---
seating_size = config.NUM_STUDENTS
relations_size = config.NUM_STUDENTS * config.NUM_STUDENTS
input_size = seating_size + relations_size # 930
action_size = len(list(itertools.combinations(range(config.NUM_STUDENTS), 2))) # 435

agent = Agent(
    seating_size=seating_size,
    relations_size=relations_size,
    action_size=action_size,
    memory_capacity=MEMORY_CAPACITY,
    learning_rate=1e-3,
    gamma=GAMMA
)
print(f"Agent初期化完了 (入力: {agent.actual_input_size} / 行動: {agent.action_size})")

# --- 2. メインの学習ループ ---
temperature = TEMP_START
scores_deque = deque(maxlen=100)
all_scores = []
start_time = time.time()

print("学習ループを開始します...")
for episode in range(1, NUM_EPISODES + 1):

    # --- (a) このエピソードで使う「ダミーのユーザー評価」をランダムに生成 ---
    # (29要素の、-1.0 から +1.0 の間のランダムな値)
    random_user_evals = (np.random.rand(config.NUM_STUDENTS - 1) * 2 - 1).astype(np.float32)

    # --- (b) 「動的マトリクス(30x30)」を生成 ★★★
    # (CSVから読んだ固定データ + ランダムなユーザー評価)
    dynamic_matrix = matrix_utils.build_full_matrix(random_user_evals, fixed_data)
    dynamic_matrix_flat = dynamic_matrix.flatten() # 900要素 (Agentに渡す用)
    
    # --- (c) 「動的マトリクス」を使って Env を初期化 ---
    env = Env(relations_matrix=dynamic_matrix)

    # 環境をリセット (6, 5) の座席表
    seating_array_2d = env.reset()

    for step in range(MAX_STEPS):

        # --- 3. 930要素の「完全な状態」を構築 ---
        seating_array_1d = seating_array_2d.flatten() # 30要素
        # [座席表(30)] + [関係性(900)] を連結
        full_state = np.concatenate((seating_array_1d, dynamic_matrix_flat))
        
        # --- 4. Agentに行動を選択させる (930要素を渡す) ---
        action_index = agent.act(full_state, temperature)
        action_pair = agent.action_pairs[action_index]

        # --- 5. 環境を1ステップ進める ---
        next_seating_array_2d, reward, done = env.step(seating_array_2d, action_pair)

        # --- 6. 930要素の「次の完全な状態」を構築 ---
        next_seating_array_1d = next_seating_array_2d.flatten()
        next_full_state = np.concatenate((next_seating_array_1d, dynamic_matrix_flat))

        # --- 7. メモリに経験を追加 (930要素のstateを記憶) ---
        agent.add_experience(full_state, action_index, reward, next_full_state, done)

        # --- 8. 学習 ---
        agent.learn(BATCH_SIZE)

        seating_array_2d = next_seating_array_2d
        if done:
            break
            
    # エピソード終了
    final_score = env.calculate_score(seating_array_2d)
    scores_deque.append(final_score)
    all_scores.append(final_score)
    temperature = max(TEMP_END, TEMP_DECAY * temperature)

    # 10エピソードごとに進捗を表示
    if episode % 10 == 0:
        avg_score = np.mean(scores_deque)
        elapsed = time.time() - start_time
        print(f"Epi: {episode}/{NUM_EPISODES} | AvgScore(100): {avg_score:.2f} | Temp: {temperature:.2f} | Time: {elapsed:.1f}s")

print("\n--- 学習が完了しました ---")

# 学習が完了したAgentの脳を保存
MODEL_PATH = "qnetwork_limited_v1.pth"
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"学習済みモデルを '{MODEL_PATH}' として保存しました。")