# api/services/change_seats/evaluate.py (★仕様変更版★)

import torch
import numpy as np
import pandas as pd
import itertools
import sys
from typing import Dict # ★ Dict をインポート

# --- (モジュールインポート try...except は変更なし) ---
try:
    from .env import Env
    from .agent import Agent
    from . import config
    from . import matrix_utils
    from .qnetwork import QNetwork
    from .replay_memory import ReplayMemory
except ImportError:
    print("--- (import error handler) ---")
    # ... (from env import Env etc.)

# --- (設定 MODEL_PATH, USER_ID は変更なし) ---
MODEL_PATH = "qnetwork_limited_v1.pth"
USER_ID = config.USER_ID # 29
NUM_SWAPS = 20 # ★ 最適化のための交換回数を定義 ★

# --- (_load_roster, _map_ids_to_names は変更なし) ---
def _load_roster() -> Dict[int, str]:
    # ... (名簿を {id: name} 辞書で返す) ...
    try:
        df_roster = pd.read_csv(config.ROSTER_PATH)
        roster_map_id_to_name = df_roster.set_index('id')['name'].to_dict()
        print("名簿(id->name)のロード完了。")
        return roster_map_id_to_name
    except Exception as e:
        print(f"警告: 名簿(id->name)ロード失敗: {e}")
        return {}

def _create_name_to_id_map(roster_map_id_to_name: Dict[int, str]) -> Dict[str, int]:
    """{id: name} 辞書から {name: id} 辞書を作成"""
    return {name: id for id, name in roster_map_id_to_name.items()}


def _map_ids_to_names(seating_array_2d: np.ndarray, roster_map_id_to_name: dict) -> list:
    """(5, 6) のID配列を (5, 6) の名前リストに変換する"""
    rows, cols = seating_array_2d.shape
    name_layout = [["" for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            student_id = int(seating_array_2d[r, c])
            name_layout[r][c] = roster_map_id_to_name.get(student_id, f"ID:{student_id}")
    return name_layout


# --- ★★★ get_recommendation を大幅修正 ★★★ ---
def get_optimized_layout(user_evaluations_dict: Dict[str, float]) -> dict:
    """
    ユーザー評価(名前:スコア)を受け取り、AIで20回席替えを
    シミュレートした後の最終配置とスコアを返す。
    """
    print(f"--- 最適化実行 (モデル: {MODEL_PATH}, 交換回数: {NUM_SWAPS}) ---")

    # --- 0. 名簿をロードし、名前->IDマップを作成 ---
    roster_map_id_to_name = _load_roster()
    roster_map_name_to_id = _create_name_to_id_map(roster_map_id_to_name)
    if not roster_map_id_to_name or not roster_map_name_to_id:
        msg = "名簿ファイルの読み込みに失敗したため、処理を中断します。"
        print(f"エラー: {msg}")
        return {"status": "error", "message": msg}

    # --- 1. ユーザー入力(辞書)をNumpy配列(ID順)に変換 ---
    expected_names_count = config.NUM_STUDENTS - 1 # 29
    if len(user_evaluations_dict) != expected_names_count:
        msg = f"入力辞書の要素数が {expected_names_count}個ではありません。"
        print(f"エラー: {msg}")
        return {"status": "error", "message": msg}

    user_evals_np = np.zeros(expected_names_count, dtype=np.float32)
    missing_names = []
    invalid_ids = []
    for name, score in user_evaluations_dict.items():
        student_id = roster_map_name_to_id.get(name)
        if student_id is None:
            missing_names.append(name)
        elif student_id == USER_ID: # ユーザー自身(ID 29)は評価対象外
             invalid_ids.append(name)
        else:
            try:
                # ID 0〜28 に対応する位置にスコアを代入
                 user_evals_np[student_id] = float(score)
            except IndexError:
                 invalid_ids.append(name) # 0-28 以外のIDの場合
            except ValueError:
                 return {"status": "error", "message": f"スコア値が無効です: {name}={score}"}


    if missing_names:
        msg = f"名簿に存在しない名前が含まれています: {missing_names}"
        print(f"エラー: {msg}")
        return {"status": "error", "message": msg}
    if invalid_ids:
         msg = f"評価対象外の名前(ユーザー自身など)が含まれています: {invalid_ids}"
         print(f"エラー: {msg}")
         return {"status": "error", "message": msg}


    # --- 2. Agentを初期化 (930入力版) ---
    print("Agentを初期化中 (930入力)...")
    # ... (Agent初期化コードは変更なし) ...
    seating_size = config.NUM_STUDENTS
    relations_size = config.NUM_STUDENTS * config.NUM_STUDENTS
    action_size = len(list(itertools.combinations(range(config.NUM_STUDENTS), 2)))
    agent = Agent(
        seating_size=seating_size,
        relations_size=relations_size,
        action_size=action_size,
        memory_capacity=1
    )

    # --- 3. 学習済みモデルをロード ---
    try:
        agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        agent.model.eval()
        print("学習済みモデルのロード完了。")
    except FileNotFoundError:
        # ... (エラーハンドリング) ...
        msg = f"モデルファイル '{MODEL_PATH}' が見つかりません。"
        return {"status": "error", "message": msg}


    # --- 4. CSVから「固定データ」をロード ---
    try:
        fixed_data = matrix_utils.load_fixed_data_from_csv()
        print("固定データ (CSV) のロード完了。")
    except Exception as e:
        # ... (エラーハンドリング) ...
        msg = f"固定データのロードに失敗: {e}"
        return {"status": "error", "message": msg}


    # --- 5. 「本番用マトリクス」を構築 ---
    production_matrix = matrix_utils.build_full_matrix(user_evals_np, fixed_data)
    production_matrix_flat = production_matrix.flatten()

    # --- 6. 「本番用Env」と「初期座席」を準備 ---
    env = Env(relations_matrix=production_matrix)
    seating_array_2d = env.reset() # ★★★ 常にランダムで初期化 ★★★
    initial_score = env.calculate_score(seating_array_2d)
    print(f"ランダムな初期座席を生成。スコア: {initial_score:.2f}")

    # --- 7. ★★★ 20回の席替えループ ★★★ ---
    current_seating_2d = seating_array_2d.copy()
    print(f"{NUM_SWAPS} 回の席替えシミュレーションを開始...")
    for i in range(NUM_SWAPS):
        # (a) 現在の 930 要素の状態を構築
        current_seating_1d = current_seating_2d.flatten()
        current_full_state = np.concatenate((current_seating_1d, production_matrix_flat))

        # (b) AIに最善の行動を予測させる
        with torch.no_grad():
            state_tensor = torch.from_numpy(current_full_state).float().unsqueeze(0)
            q_values = agent.model(state_tensor)
            action_index = torch.argmax(q_values).item()
        action_pair = agent.action_pairs[action_index]

        # (c) 行動を実行して座席表を更新
        next_seating_array_2d, reward, _ = env.step(current_seating_2d, action_pair)
        current_score = env.calculate_score(next_seating_array_2d) # 現在のスコアを更新

        # (d) ログ表示 (オプション)
        s1_name = roster_map_id_to_name.get(action_pair[0], f"ID:{action_pair[0]}")
        s2_name = roster_map_id_to_name.get(action_pair[1], f"ID:{action_pair[1]}")
        print(f"  Step {i+1}/{NUM_SWAPS}: 「{s1_name}」↔「{s2_name}」 (Reward: {reward:.2f} -> Score: {current_score:.2f})")

        # (e) 次のループのために座席表を更新
        current_seating_2d = next_seating_array_2d

    # --- 8. 最終結果を準備 ---
    final_score = env.calculate_score(current_seating_2d) # 20回交換後のスコア
    final_seating_names = _map_ids_to_names(current_seating_2d, roster_map_id_to_name)

    print("--- シミュレーション完了 ---")
    print(f"初期スコア: {initial_score:.2f}")
    print(f"最終スコア ({NUM_SWAPS}回交換後): {final_score:.2f}")

    # --- 9. 最終レスポンスを構築 ---
    return {
        "initial_score": float(initial_score),
        "final_layout_names": final_seating_names, # ★ 5x6 の名前リスト
        "final_score": float(final_score),
    }

# --- (if __name__ == "__main__": のテストブロックは更新が必要) ---
# --- (ここでは省略。FastAPI経由でテストしてください) ---