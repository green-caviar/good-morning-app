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

@router.get("/env")
async def env_test():
    """
    Envクラスの動作確認用エンドポイント。

    1. Env() をインスタンス化し、relationships.csv を読み込む。
    2. reset() を呼び出し、初期座席表を生成する。
    3. calculate_score() で初期スコアを計算する。

    これらが成功すれば、CSVの読み込みとEnvの基本ロジックが
    正しく機能していることを確認できます。
    """
    try:
        # 1. Envのインスタンスを作成
        #    (この瞬間に 'relationships.csv' が読み込まれます)
        env = Env()

        # 2. reset() をテスト
        initial_state = env.reset()

        # 3. calculate_score() をテスト
        #    (内部で self.relations を参照します)
        initial_score = env.calculate_score(initial_state)

        # 4. 成功したことをSwaggerUIに返す
        return {
            "status": "success",
            "message": "Env initialized and tested successfully.",
            "csv_shape": env.relations.shape,
            "initial_state_shape": initial_state.shape,
            "initial_score": initial_score,
            # "initial_state_sample": initial_state.tolist()[0] # 最初の行だけサンプルで返す
        }
    except FileNotFoundError as e:
        # env.py がCSVを見つけられなかった場合
        return {
            "status": "error",
            "message": "FileNotFoundError: 'relationships.csv' が見つかりません。",
            "details": str(e),
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        # その他の予期せぬエラー
        return {
            "status": "error",
            "message": "Envのテスト中に予期せぬエラーが発生しました。",
            "error_type": type(e).__name__,
            "details": str(e),
            "traceback": traceback.format_exc()
        }

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


# --- 定数 ---
SEATING_SIZE = 30  # 30席
RELATIONS_SIZE = 900 # 30x30 マトリクス
INPUT_SIZE = SEATING_SIZE + RELATIONS_SIZE # 930

# 30人から2人を選ぶ組み合わせ = 435通り
ACTION_SIZE = len(list(itertools.combinations(range(SEATING_SIZE), 2))) 

# --- グローバル変数としてAgentを初期化 ---
print("FastAPIサーバー起動中...")
print(f"Agentを {INPUT_SIZE} 入力で初期化します...")

agent = Agent(
    seating_size=SEATING_SIZE,
    relations_size=RELATIONS_SIZE,
    action_size=ACTION_SIZE,
    memory_capacity=1000 # テスト用なので小さくてOK
)
print("Agent初期化完了。")


# --- 入力データの型定義 ---
class StateInput(BaseModel):
    # conlist は「要素数が固定されたリスト」を定義します
    state: conlist(float, min_length=INPUT_SIZE, max_length=INPUT_SIZE)

# --- エンドポイントの定義 ---

@router.get("/")
def read_root():
    return {"message": "席替え最適化AI (DQN) 930入力テストサーバー"}

@router.post("/act")
def get_action(state_input: StateInput, temperature: float = 0.1):
    """
    930要素の状態ベクトルを受け取り、
    AIが判断した「次の行動(action_index)」を返します。
    """
    
    # 1. PydanticモデルからNumpy配列に変換
    # (930,) のNumpy配列が完成
    state_vector = np.array(state_input.state, dtype=np.float32)

    # 2. 修正した agent.act を呼び出し
    try:
        action_index = agent.act(state_vector, temperature)
        
        # 3. action_index を人間がわかる「ペア」に翻訳
        action_pair = agent.action_pairs[action_index]
        
        return {
            "message": "Action calculated successfully.",
            "input_state_shape": state_vector.shape,
            "action_index": action_index,
            "action_pair (student_id_1, student_id_2)": action_pair
        }
    except Exception as e:
        return {"error": f"Agentの実行中にエラーが発生しました: {e}"}
# AIの設定 (train.pyと一致させる)
STATE_SIZE = 30
ACTION_SIZE = len(list(itertools.combinations(range(STATE_SIZE), 2))) # 435
MODEL_PATH = "data/qnetwork.pth" # 学習済みモデルのパス

def initialize_agent():
    """学習済みモデルをロードしたAgentインスタンスを作成する"""
    
    # 1. Agentの器を作成 (メモリは推論時には不要)
    agent = Agent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memory_capacity=1 # 推論時はmemoryを使わない
    )
    
    # 2. 学習済みの「脳（重み）」をロードする
    if not os.path.exists(MODEL_PATH):
        print(f"致命的エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        # FastAPIの起動を失敗させる
        raise FileNotFoundError(f"モデルファイル '{MODEL_PATH}' がありません。train.py を実行してください。")

    # GPUがない環境でもエラーなくロードするため 'cpu' を指定
    agent.model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )
    
    # 3. 必ず「予測モード (evaluation mode)」に切り替える (重要)
    # これにより、勾配計算などがオフになり、高速・安全に予測できる
    # (※ agent.act() 内部でも自動で切り替わりますが、明示的に行うのが安全です)
    agent.model.eval() 
    
    print(f"'{MODEL_PATH}' から学習済みモデルをロードしました。[推論モード]")
    return agent

# FastAPI起動時に、EnvとAgentをそれぞれ1回だけ初期化
try:
    # Envも初期化しておき、スコア計算に使えるようにする
    loaded_env = Env()
    loaded_agent = initialize_agent()
except FileNotFoundError as e:
    print(f"起動時エラー: {e}")
    # CSVやPTHがない場合は起動を停止させる
    exit()

# FastAPIのDI(Dependency Injection)を使い、
# エンドポイント実行時にAgentインスタンスを「注入」する
def get_agent():
    """DI用の関数: ロード済みのAgentインスタンスを返す"""
    return loaded_agent

def get_env():
    """DI用の関数: ロード済みのEnvインスタンスを返す"""
    return loaded_env


# (router = APIRouter() は定義済みと仮定)
router = APIRouter()
# (もし router がなければこの行のコメントを外す → router = APIRouter())


# === 4. リクエスト/レスポンス用のPydanticモデル定義 ===

class SeatLayout(BaseModel):
    # [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], ...] のような6x5の配列を期待
    layout: list[list[int]]
    model_config = {
        "example": {
            "layout": [
                [ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29]
            ]
        }
    }


class SwapSuggestion(BaseModel):
    suggested_pair: list[int] # 例: [1, 5]
    temperature_used: float
    current_score: float # (おまけ: 現在のスコアも返す)
    model_config = {
        "example": {
            "suggested_pair": [1, 5],
            "temperature_used": 1.0,
            "current_score": 31.6
        }
    }


# === 5. 席替え推薦のエンドポイント (これがAPI本体) ===

@router.post("/suggest_swap", response_model=SwapSuggestion)
async def suggest_swap(
    seat_layout: SeatLayout,
    temperature: float = Query(1.0, ge=0.01, description="席替えのランダム性（温度）。0.1で堅実、1.0でバランス、3.0で大胆。"),
    agent: Agent = Depends(get_agent),
    env: Env = Depends(get_env)
):
    """
    現在の座席表(6x5)を受け取り、学習済みAIが最適と判断した
    「次の一手（交換ペア）」を提案します。
    """
    
    # 1. リクエストボディ(2D List)をNumPy配列(2D)に変換
    state_2d = np.array(seat_layout.layout)

    # 2. 形状チェック (安全のため)
    if state_2d.shape != (6, 5):
        raise HTTPException(status_code=400, detail=f"座席表の形状が(6, 5)ではありません。Shape: {state_2d.shape}")

    # 3. AIに渡すために1D (30,)にフラット化
    state_1d = state_2d.flatten()

    # 4. [Env] 現在のスコアを計算 (おまけ機能)
    current_score = env.calculate_score(state_2d)

    # 5. [Agent] AIに行動を決定させる (予測モード)
    #    agent.act() は内部で model.eval() と torch.no_grad() を行う
    action_index = agent.act(state_1d, temperature=temperature)
    
    # 6. index を IDペアに「翻訳」
    action_pair = agent.action_pairs[action_index]
    
    # 7. 結果をJSONで返す
    return SwapSuggestion(
        suggested_pair=list(action_pair), # (1, 5) -> [1, 5]
        temperature_used=temperature,
        current_score=current_score
    )

# (もし api/main.py に直接書いているなら、app に router を含める)
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(router, prefix="/api/v1") # 例

# === 4. レスポンス用のPydanticモデル定義 (★新規追加) ===

class OptimizedLayoutResponse(BaseModel):
    final_layout: list[list[int]] # 最終的な座席表 (6x5)
    initial_score: float        # 初期スコア
    final_score: float          # 最終スコア
    steps_taken: int            # AIが実行した交換回数
    temperature_used: float
    model_config = {
        "example": {
            "final_layout": [
                [ 10, 2, 5, 29, 14],
                [ 21, 13, 7, 8, 9],
                # ... (以下略) ...
                [ 25, 1, 27, 28, 4]
            ],
            "initial_score": 12.0,
            "final_score": 58.0,
            "steps_taken": 100,
            "temperature_used": 0.1
        }
    }


# === 5. AIおまかせ席替えエンドポイント (★これが新しいAPI) ===

@router.get("/get_optimized_layout", response_model=OptimizedLayoutResponse)
async def get_optimized_layout(
    steps: int = Query(100, ge=1, le=500, description="AIが最適化のために試行する交換回数（ステップ数）"),
    temperature: float = Query(0.1, ge=0.01, description="AIの行動の堅実さ（温度）。0.1が最も堅実（活用）。"),
    agent: Agent = Depends(get_agent),
    env: Env = Depends(get_env)
):
    """
    AIによる席替えの最適化を実行し、
    「最終的な座席表」を返します。
    
    (入力は不要です)
    """
    
    # 1. ランダムな初期座席表を生成 (2D)
    state_2d = env.reset()
    initial_score = env.calculate_score(state_2d)

    # 2. AIによる最適化ループを実行
    for _ in range(steps):
        
        # 3. AIに渡すために1Dに変換
        state_1d = state_2d.flatten()
        
        # 4. AIに行動を決定させる (予測モード)
        #    指定された低温(堅実モード)で実行
        action_index = agent.act(state_1d, temperature=temperature)
        action_pair = agent.action_pairs[action_index]
        
        # 5. 環境を1ステップ進める (env.step は 2D配列を要求)
        next_state_2d, reward, done = env.step(state_2d, action_pair)
        
        # 6. 座席表を更新
        state_2d = next_state_2d

    # 7. ループ完了後、最終スコアを計算
    final_score = env.calculate_score(state_2d)
    
    # 8. 最終結果をJSONで返す
    return OptimizedLayoutResponse(
        final_layout=state_2d.tolist(), # NumPy配列(2D)をリスト(2D)に変換
        initial_score=initial_score,
        final_score=final_score,
        steps_taken=steps,
        temperature_used=temperature
    )
