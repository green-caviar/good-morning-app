from fastapi import APIRouter, Query, Depends, HTTPException
from typing import List
from api.services.change_seats.env import Env
from api.services.change_seats.agent import Agent
from api.schemas.change_seats import Change_seats
import traceback
import numpy as np
import itertools
import time

from pydantic import BaseModel, conlist
import os
import torch


router = APIRouter()

@router.get("/env-step2")
async def env_step2_test():
    """
    ステップ2で修正した Env クラスの動作確認用エンドポイント。

    1. 30x30 のダミーの関係性マトリクスを Numpy で作成する。
    2. Env(relations_matrix=...) にダミーマトリクスを渡してインスタンス化する。
    3. reset() を呼び出し、初期座席表を生成する。
    4. calculate_score() で初期スコアを計算する。

    これらが成功すれば、Envが外部マトリクスを正しく受け取れていることを
    確認できます。
    """
    try:
        # --- 1. ダミーの関係性マトリクスを作成 ---
        NUM_STUDENTS = 30
        # 全員が「普通(0)」の関係性を持つダミーマトリクス
        dummy_matrix = np.zeros((NUM_STUDENTS, NUM_STUDENTS), dtype=np.float32)

        # --- 2. 修正版 Env をインスタンス化 ---
        # (この瞬間にダミーマトリクスが self.relations に格納される)
        env = Env(relations_matrix=dummy_matrix)

        # --- 3. reset() をテスト ---
        initial_state = env.reset() # (6, 5) の配列が返るはず

        # --- 4. calculate_score() をテスト ---
        # (内部でダミーマトリクス(全部0)が参照される)
        initial_score = env.calculate_score(initial_state)

        # 5. 成功したことをSwaggerUIに返す
        return {
            "status": "success",
            "message": "Env (Step 2) initialized and tested successfully.",
            "passed_matrix_shape": env.relations.shape,  # (タプルはJSONセーフ)
            "initial_state_shape": initial_state.shape,  # (タプルはJSONセーフ)
            
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # 修正点: numpy.float32 を Python の float に変換
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            "initial_score": float(initial_score)
        }
    except ValueError as e:
        # Envの形状チェックでエラーが起きた場合
        return {
            "status": "error",
            "message": "Envの初期化中に形状エラーが発生しました。",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        # その他の予期せぬエラー
        return {
            "status": "error",
            "message": "Env (Step 2) のテスト中に予期せぬエラーが発生しました。",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc()
        }

# --- ここまで追加 ---

@router.get("/agent")
async def agent_test():
    """
    Agentクラスの動作確認用エンドポイント。

    1. Agent() を state_size=30 でインスタンス化する。
       (内部で QNetwork(30, 435) と ReplayMemory が初期化される)
    2. ダミーの state (30要素) で agent.act() を実行する。
    3. agent.add_experience() でメモリに経験を追加する。

    これらが成功すれば、Agentと関連クラスが正しく連携していることを確認できます。
    """
    try:
        # --- 1. Agentの初期設定を定義 ---
        STATE_SIZE = 30 # 30人 (6x5)

        # 行動の総数 = 30人から2人を選ぶ組み合わせ
        # (30 * 29) / 2 = 435 通り
        ACTION_SIZE = len(list(itertools.combinations(range(STATE_SIZE), 2)))

        if ACTION_SIZE != 435:
            # 基本的な計算が間違っていないかチェック
             return {
                "status": "error",
                "message": f"ACTION_SIZE の計算が 435 ではありません。 (計算結果: {ACTION_SIZE})"
             }

        MEMORY_CAPACITY = 100 # テスト用の小さなメモリ

        # --- 2. Agentのインスタンスを作成 ---
        # (この瞬間に QNetwork(30, 435) と ReplayMemory(100) が作られる)
        agent = Agent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            memory_capacity=MEMORY_CAPACITY,
            learning_rate=0.001,
            gamma=0.99
        )

        # --- 3. ダミーの state (座席表) を作成 ---
        # (30,) の形状のNumPy配列 (0-29のシャッフル)
        dummy_state = np.random.permutation(STATE_SIZE)

        # --- 4. agent.act() をテスト ---
        # (30,) のNumPy配列を渡す
        action_index = agent.act(dummy_state, temperature=1.0) # T=1.0でランダム性を高める

        # --- 5. agent.add_experience() をテスト ---
        dummy_next_state = np.random.permutation(STATE_SIZE)
        dummy_reward = 1.0
        dummy_done = False

        agent.add_experience(
            dummy_state,
            action_index,
            dummy_reward,
            dummy_next_state,
            dummy_done
        )

        # --- 6. 成功したことをSwaggerUIに返す ---
        return {
            "status": "success",
            "message": "Agent initialized and tested successfully.",
            "tested_methods": ["__init__", "act", "add_experience"],
            "config": {
                "state_size_in": agent.num_students, # 30
                "action_size_in": agent.action_size,  # 435
                # QNetworkの入力層が正しく 30 になっているか確認
                "q_network_input_features": agent.model.layer1.in_features 
            },
            "act_result": {
                "action_index_received": action_index,
                "is_valid_index": (0 <= action_index < ACTION_SIZE)
            },
            "memory_result": {
                "memory_length": len(agent.memory) # 1 が返るはず
            }
        }

    except ImportError as e:
        return {
            "status": "error",
            "message": "ImportError: Agent, QNetwork, ReplayMemory の import に失敗しました。",
            "details": str(e),
            "traceback": traceback.format_exc(),
            "hint": "api/main.py から見て、agent.py 等のパスが正しいか確認してください (例: from ..agent import Agent)"
        }
    except Exception as e:
        # その他の予期せぬエラー (例: QNetworkの初期化ミスなど)
        return {
            "status": "error",
            "message": "Agentのテスト中に予期せぬエラーが発生しました。",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc()
        }
# --- ここまで追加 ---

@router.get("/env-agent-step")
async def env_agent_step_test():
    """
    EnvとAgentを連携させ、学習の「1ステップ」を実行するテスト。

    1. Env と Agent の両方をインスタンス化する。
    2. [Env]  env.reset() で初期座席表(2D)を取得 -> 1Dに変換。
    3. [Agent] agent.act(state_1d) で行動(index)を決定。
    4. [Env]  env.step(state_2d, action_pair) で環境を1ステップ進める。
    5. [Agent] agent.add_experience(...) で結果をメモリに保存する。

    これが成功すれば、EnvとAgentの間のデータ連携（握手）が
    正しく行えていることが確認できます。
    """
    try:
        # --- 1. Agent のための基本設定 ---
        STATE_SIZE = 30 # 30人
        ACTION_SIZE = len(list(itertools.combinations(range(STATE_SIZE), 2))) # 435
        MEMORY_CAPACITY = 100

        # --- 2. 両方のクラスをインスタンス化 ---
        env = Env()
        agent = Agent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            memory_capacity=MEMORY_CAPACITY
        )

        # --- 3. [Env] 初期状態を取得 ---
        # state_2d は (6, 5) の NumPy 配列
        state_2d = env.reset()

        # [変換] Agent に渡すために 1D (30,) にフラット化
        state_1d = state_2d.flatten()

        # --- 4. [Agent] 行動を決定 ---
        # agent.act は 1D (30,) の配列を期待する
        action_index = agent.act(state_1d, temperature=1.0)

        # [変換] action_index を Env が理解できる IDペア (e.g., (5, 12)) に変換
        action_pair = agent.action_pairs[action_index]

        # --- 5. [Env] 環境を1ステップ進める ---
        # env.step は 2D (6, 5) の座席表 と IDペア を期待する
        next_state_2d, reward, done = env.step(state_2d, action_pair)

        # [変換] Agent のメモリに保存するために 1D (30,) にフラット化
        next_state_1d = next_state_2d.flatten()

        # --- 6. [Agent] 経験をメモリに保存 ---
        # メモリには 1D (30,) の state / next_state を保存する
        agent.add_experience(
            state_1d,       # (30,)
            action_index,   # 整数
            reward,         # 浮動小数点数
            next_state_1d,  # (30,)
            done            # ブール値
        )

        # --- 7. 成功したことをSwaggerUIに返す ---
        return {
            "status": "success",
            "message": "Env and Agent stepped successfully together.",
            "0_env_reset_shape": state_2d.shape,
            "1_agent_act_input_shape": state_1d.shape,
            "2_agent_act_output": {
                "action_index": action_index,
                "action_pair": action_pair
            },
            "3_env_step_output": {
                "next_state_shape": next_state_2d.shape,
                "reward": reward,
                "done": done
            },
            "4_agent_memory_input_shape": next_state_1d.shape,
            "5_agent_memory_length": len(agent.memory) # 1
        }

    except ImportError as e:
        return {
            "status": "error",
            "message": "ImportError: Env または Agent の import に失敗しました。",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }
    except Exception as e:
        # EnvとAgentの間のデータ形状が間違っている場合など
        return {
            "status": "error",
            "message": "EnvとAgentの連携ステップ中に予期せぬエラーが発生しました。",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Envが返す形状(6,5)とAgentが期待する形状(30,)の .flatten() 変換が正しいか確認してください。"
        }

# エンドポイントの実行がタイムアウトしないよう、
# テスト用の小さな値を設定します。
TEST_EPISODES = 10       # 10回のシミュレーションのみ実行
TEST_MAX_STEPS = 20      # 1シミュレーションあたり20回のみ席交換
TEST_BATCH_SIZE = 8      # 学習時のバッチサイズも小さめに
TEST_MEMORY_CAPACITY = 1000 # メモリも小さめ

@router.get("/train")
async def train_test():
    """
    train.py の学習ループ全体の動作確認用エンドポイント。
    
    1. Env と Agent を初期化する。
    2.「テスト用の少ない回数」(10エピソード x 20ステップ) で
       学習ループ (act -> step -> add_experience -> learn) を実行する。
    3. ループがエラーなく完了し、スコアが記録されるか確認する。
    """
    start_time = time.time()
    try:
        # --- 1. train.py の初期化ロジック ---
        env = Env()
        
        NUM_STUDENTS = 30
        ACTION_SIZE = len(list(itertools.combinations(range(NUM_STUDENTS), 2))) # 435

        agent = Agent(state_size=NUM_STUDENTS,
                      action_size=ACTION_SIZE,
                      memory_capacity=TEST_MEMORY_CAPACITY, # Test
                      learning_rate=1e-3,
                      gamma=0.99)
        
        # 温度設定 (train.py と同じものを使用)
        TEMP_START = 2.0
        TEMP_END = 1.0
        TEMP_DECAY = 0.995
        temperature = TEMP_START
        
        all_scores = [] # スコア履歴

        # --- 2. train.py のメインループ (テスト回数版) ---
        for episode in range(1, TEST_EPISODES + 1):
            
            state_array = env.reset() # (6, 5) 2D
            
            for step in range(TEST_MAX_STEPS): # 20ステップ
                
                # 1. 行動 (Env(2D) -> Agent(1D))
                state_1d = state_array.flatten()
                action_index = agent.act(state_1d, temperature)
                action_pair = agent.action_pairs[action_index]
                
                # 2. 実行 (Env)
                next_state_array, reward, done = env.step(state_array, action_pair)
                
                # 3. 記憶 (Agent(1D))
                next_state_1d = next_state_array.flatten()
                agent.add_experience(state_1d, action_index, reward, next_state_1d, done)
                
                # 4. 学習 (Agent)
                agent.learn(TEST_BATCH_SIZE) # Test
                
                state_array = next_state_array
                
                if done:
                    break
            
            # エピソード終了
            final_score = env.calculate_score(state_array)
            all_scores.append(final_score)
            
            # 温度更新
            temperature = max(TEMP_END, TEMP_DECAY * temperature)

        # --- 3. ループ完了。結果を返す ---
        end_time = time.time()
        
        return {
            "status": "success",
            "message": "Training loop test completed successfully.",
            "config": {
                "episodes_run": TEST_EPISODES,
                "steps_per_episode": TEST_MAX_STEPS
            },
            "timing_seconds": (end_time - start_time),
            "results": {
                "final_scores_history": all_scores,
                "average_score": np.mean(all_scores),
                "final_temperature": temperature
            },
            "memory_status": {
                "length": len(agent.memory), # (10 * 20) = 200 になるはず
                "capacity": TEST_MEMORY_CAPACITY
            }
        }

    except ImportError as e:
        return {
            "status": "error",
            "message": "ImportError: Env または Agent の import に失敗しました。",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }
    except Exception as e:
        # ループの途中でエラーが起きた場合 (例: learnメソッドのエラー)
        return {
            "status": "error",
            "message": "学習ループの実行中に予期せぬエラーが発生しました。",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc(),
            "hint": "agent.learn() や agent.add_experience() の内部ロジックを確認してください。"
        }
# --- ここまで追加 ---

try:
    from api.services.change_seats.agent import Agent
except ImportError:
    print("="*50)
    print("エラー: agent.py が見つかりません。")
    print("test_fastapi.py と同じ階層（または適切なパス）に配置してください。")
    print("="*50)
    exit()


# # --- 定数 ---
# STATE_SIZE = 30  # 30席
# RELATIONS_SIZE = 900 # 30x30 マトリクス
# INPUT_SIZE = STATE_SIZE + RELATIONS_SIZE # 930

# # 30人から2人を選ぶ組み合わせ = 435通り
# ACTION_SIZE = len(list(itertools.combinations(range(STATE_SIZE), 2))) 

# # --- グローバル変数としてAgentを初期化 ---
# print("FastAPIサーバー起動中...")
# print(f"Agentを {INPUT_SIZE} 入力で初期化します...")

# agent = Agent(
#     state_size=STATE_SIZE,
#     relations_size=RELATIONS_SIZE,
#     action_size=ACTION_SIZE,
#     memory_capacity=1000 # テスト用なので小さくてOK
# )
# print("Agent初期化完了。")


# # --- 入力データの型定義 ---
# class StateInput(BaseModel):
#     # conlist は「要素数が固定されたリスト」を定義します
#     state: conlist(float, min_length=INPUT_SIZE, max_length=INPUT_SIZE)

# # --- エンドポイントの定義 ---

# @router.get("/")
# def read_root():
#     return {"message": "席替え最適化AI (DQN) 930入力テストサーバー"}

# @router.post("/act")
# def get_action(state_input: StateInput, temperature: float = 0.1):
#     """
#     930要素の状態ベクトルを受け取り、
#     AIが判断した「次の行動(action_index)」を返します。
#     """
    
#     # 1. PydanticモデルからNumpy配列に変換
#     # (930,) のNumpy配列が完成
#     state_vector = np.array(state_input.state, dtype=np.float32)

#     # 2. 修正した agent.act を呼び出し
#     try:
#         action_index = agent.act(state_vector, temperature)
        
#         # 3. action_index を人間がわかる「ペア」に翻訳
#         action_pair = agent.action_pairs[action_index]
        
#         return {
#             "message": "Action calculated successfully.",
#             "input_state_shape": state_vector.shape,
#             "action_index": action_index,
#             "action_pair (student_id_1, student_id_2)": action_pair
#         }
#     except Exception as e:
#         return {"error": f"Agentの実行中にエラーが発生しました: {e}"}
    
# # あなたの router.py に追加するテストエンドポイント (修正版)

# from api.services.change_seats import config
# from api.services.change_seats import matrix_utils

# # あなたの router.py の @router.get("/step3-matrix-utils") を置き換え

# # (config と matrix_utils のインポートは既にある前提)
# # (import numpy as np, traceback も既にある前提)

# @router.get("/step3-matrix-utils")
# async def step3_matrix_utils_test():
#     """
#     ステップ3(CSV対応版)の動作確認。
    
#     1. matrix_utils.load_fixed_data_from_csv() で (29, 29) の固定データをロード
#     2. ダミーの「ユーザー評価」(29要素) を作成
#     3. matrix_utils.build_full_matrix() で (30, 30) のマトリクスを生成
#     4. 生成されたマトリクスが、ルール通りか検証する
#     """
#     try:
#         # --- 1. CSVから「固定データ」をロード ---
#         # (ここで config.CSV_PATH が参照される)
#         fixed_data = matrix_utils.load_fixed_data_from_csv()

#         if fixed_data.shape != (config.NUM_STUDENTS - 1, config.NUM_STUDENTS - 1):
#             raise ValueError(f"ロードした固定データの形状が(29, 29)ではありません: {fixed_data.shape}")

#         # --- 2. ダミーの「ユーザー評価」を作成 ---
#         # (ユーザー29番から、0〜28番への評価)
#         user_evals = np.zeros(config.NUM_STUDENTS - 1, dtype=np.float32) # 29要素
#         user_evals[0] = 1.0  # (29 -> 0 への評価)
#         user_evals[1] = -1.0 # (29 -> 1 への評価)

#         # --- 3. マトリクス生成関数を呼び出し ---
#         full_matrix = matrix_utils.build_full_matrix(user_evals, fixed_data)

#         # --- 4. 検証 ---
#         USER_ID = config.USER_ID # 29
        
#         # (a) 形状は 30x30 か？
#         if full_matrix.shape != (config.NUM_STUDENTS, config.NUM_STUDENTS):
#             raise ValueError(f"マトリクスの形状が(30, 30)ではありません: {full_matrix.shape}")

#         # (b) ユーザーの「行」は正しく反映されたか？
#         test_val_1 = full_matrix[USER_ID, 0]
#         test_val_2 = full_matrix[USER_ID, 1]
#         if test_val_1 != 1.0 or test_val_2 != -1.0:
#             raise ValueError(
#                 f"ユーザー評価(行)が正しく反映されていません。"
#                 f"TestVal1: {test_val_1}, TestVal2: {test_val_2}"
#             )

#         # (c) ユーザーの「列」は全部 0 (固定) か？
#         user_col_sum = np.sum(full_matrix[:, USER_ID])
#         if user_col_sum != 0.0:
#             raise ValueError(f"ユーザー列(他者->自分)が 0 で固定されていません。")
        
#         # (d) 固定データは正しく反映されたか？ (0,0)地点を比較
#         if full_matrix[0, 0] != fixed_data[0, 0]:
#              raise ValueError("固定データ(0,0)が正しく反映されていません。")

#         # --- 5. 成功 ---
#         return {
#             "status": "success",
#             "message": "matrix_utils (CSV対応版) は正常に動作しています。",
#             "loaded_fixed_data_shape": fixed_data.shape,
#             "built_matrix_shape": full_matrix.shape,
#             "validation_checks": {
#                 "fixed_data_check": "OK",
#                 "user_row_check": "OK",
#                 "user_col_check (fixed_zero)": "OK"
#             }
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": "matrix_utils (CSV対応版) のテスト中にエラーが発生しました。",
#             "error_type": type(e).__name__,
#             "details": str(e),
#             "traceback": traceback.format_exc()
#         }

# api/routers/change_seats.py

from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel, conlist, Field # ★ Pydantic をインポート
import numpy as np
import traceback

# --- 必要なモジュールをインポート ---
from api.services.change_seats.env import Env
from api.services.change_seats.agent import Agent
from api.services.change_seats import config
from api.services.change_seats import matrix_utils

# ★ 1. 評価モジュールをインポート ★
try:
    from api.services.change_seats import evaluate
except ImportError as e:
    print(f"CRITICAL: evaluate.py のインポートに失敗しました: {e}")
    evaluate = None

# ★ 2. 入力用のPydanticモデルを定義 ★
class UserEvaluationInput(BaseModel):
    # 29個の浮動小数点数のリストを要求
    user_evals: conlist(float, min_length=29, max_length=29)
    # (オプション) 初期座席表を指定できるようにする
    initial_seating: Optional[List[List[int]]] = None

router = APIRouter()

# ... ( /env, /agent, /step3-matrix-utils などのテストエンドポイント ... )
# ... (これらは本番では不要なら削除してもOK) ...


# ★ 3. 最終的なAPIエンドポイント ★
# api/routers/change_seats.py

from fastapi import APIRouter
from typing import List, Optional, Dict # ★ Dict を追加
from pydantic import BaseModel, conlist, validator # ★ validator を追加
import numpy as np
import traceback
import pandas as pd

# ... (Env, Agent, config, matrix_utils, evaluate のインポート) ...

# ★ 1. 入力モデルを辞書に変更 ★
class UserEvaluationInput(BaseModel):
    # キーが文字列、値が浮動小数点数の辞書を要求
    user_evals: Dict[str, float]

    # バリデーターを追加して、要素数が29個かチェック
    @validator('user_evals')
    def check_dict_size(cls, v):
        expected_size = config.NUM_STUDENTS - 1
        if len(v) != expected_size:
            raise ValueError(f'辞書の要素数は {expected_size} 個である必要があります')
        return v
    
    # initial_seating フィールドは削除

def create_example_evals() -> Dict[str, float]:
    """名簿ファイルを読み込み、Swagger UI 用のサンプル評価辞書を作成する"""
    example_dict = {}
    try:
        df_roster = pd.read_csv(config.ROSTER_PATH)
        # ユーザー(ID=29)を除いた名前リストを取得
        names = df_roster[df_roster['id'] != config.USER_ID]['name'].tolist()
        if len(names) != config.NUM_STUDENTS - 1:
            print(f"警告: 名簿の人数({len(names)})が期待値({config.NUM_STUDENTS - 1})と異なります。")
        for name in names:
            example_dict[name] = 0.0
    except Exception as e:
        print(f"警告: サンプルデータの生成に失敗しました: {e}")
        # 失敗した場合の代替サンプル
        example_dict = {f"生徒{i}": 0.0 for i in range(config.NUM_STUDENTS - 1)}

    # 要素数が正しいか最終チェック
    if len(example_dict) != config.NUM_STUDENTS - 1:
         return {f"生徒{i}": 0.0 for i in range(config.NUM_STUDENTS - 1)}
    return example_dict

# ★ 2. 入力モデルの定義を修正 ★
class UserEvaluationInput(BaseModel):
    # Field を使って example を指定
    user_evals: Dict[str, float] = Field(
        ..., # '...' は必須フィールドを示す
        example=create_example_evals() # ★ ヘルパー関数で生成したサンプルを指定 ★
    )

    # バリデーターは変更なし
    @validator('user_evals')
    def check_dict_size(cls, v):
        expected_size = config.NUM_STUDENTS - 1
        if len(v) != expected_size:
            raise ValueError(f'辞書の要素数は {expected_size} 個である必要があります')
        return v

router = APIRouter()

# ... (古いテストエンドポイント) ...

# ★ 2. エンドポイントの修正 ★
@router.post("/recommend")
async def recommend_seat_change(input_data: UserEvaluationInput): # 引数を新しいモデルに
    """
    ユーザー評価(名前:スコアの辞書)を受け取り、
    AIで20回席替えをシミュレートした後の最終配置とスコアを返す。

    最終レスポンス:
    - final_layout_names: (5x6) の名前入り座席表
    - final_score: 席替え後のスコア
    """
    if evaluate is None:
        return {"status": "error", "message": "サーバーエラー: evaluateモジュールがロードされていません。"}

    try:
        # initial_seating の処理は不要になったので削除

        # 新しい関数名を呼び出し、辞書を渡す
        result = evaluate.get_optimized_layout(
            user_evaluations_dict=input_data.user_evals # リストではなく辞書を渡す
            # initial_seating_array は渡さない
        )

        return result

    except Exception as e:
        # ... (エラーハンドリングは同じ) ...
        return {
            "status": "error",
            "message": "評価の実行中に予期せぬエラーが発生しました。",
            # ...
        }