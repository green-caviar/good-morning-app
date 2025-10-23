from fastapi import APIRouter
from typing import List
from api.services.change_seats.env import Env
from api.schemas.change_seats import Change_seats
import traceback

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