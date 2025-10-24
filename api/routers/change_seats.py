from fastapi import APIRouter
from typing import List
from api.services.change_seats.env import Env
from api.services.change_seats.agent import Agent
from api.schemas.change_seats import Change_seats
import traceback
import numpy as np
import itertools
import time

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