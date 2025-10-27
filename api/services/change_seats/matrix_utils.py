# api/services/change_seats/matrix_utils.py (CSV対応版)

import numpy as np
import pandas as pd
from . import config 

def load_fixed_data_from_csv() -> np.ndarray:
    """
    config.CSV_PATH からCSVを読み込み、
    「固定データ (29x29)」を抽出して返す。
    
    戻り値:
        (29, 29) の固定データ (Numpy配列)
    """
    try:
        # 1. CSVを読み込む (header=0, index_col=0 が (30,30) だった)
        df = pd.read_csv(config.CSV_PATH, header=0, index_col=0)
        full_matrix_data = df.values.astype(np.float32)
        
        if full_matrix_data.shape != (config.NUM_STUDENTS, config.NUM_STUDENTS):
             raise ValueError(f"CSVの形状が(30,30)ではありません。Shape: {full_matrix_data.shape}")
             
        # 2. 「固定データ」を抽出
        # ユーザーID(29) を除く、0〜28行目、0〜28列目の部分
        USER_ID = config.USER_ID
        fixed_data = full_matrix_data[:USER_ID, :USER_ID]
        
        if fixed_data.shape != (config.NUM_STUDENTS - 1, config.NUM_STUDENTS - 1):
             raise ValueError(f"固定データ(29,29)の抽出に失敗しました。Shape: {fixed_data.shape}")
             
        return fixed_data

    except FileNotFoundError:
        print(f"エラー: {config.CSV_PATH} が見つかりません。")
        raise
    except Exception as e:
        print(f"matrix_utils.load_fixed_data_from_csv でエラー: {e}")
        raise


def build_full_matrix(
    user_evaluations: np.ndarray, 
    fixed_data: np.ndarray
) -> np.ndarray:
    """
    「固定データ(29x29)」と「ユーザー評価(29,)」を合体させ、
    30x30の完全な関係性マトリクスを生成する。
    
    引数:
        user_evaluations: ユーザー(29番)から他(0〜28番)への評価 (29,)
        fixed_data:       他者(0〜28番)同士の関係性 (29, 29)
    
    戻り値:
        30x30 の完全な関係性マトリクス
    """
    
    # 1. 30x30 のゼロ行列（土台）を作成
    full_matrix = np.zeros(
        (config.NUM_STUDENTS, config.NUM_STUDENTS), 
        dtype=np.float32
    )
    
    USER_ID = config.USER_ID # 29

    # 2. 「固定データ」を土台にコピー
    # (0〜28行目, 0〜28列目) の部分に (29x29) の固定データを挿入
    full_matrix[:USER_ID, :USER_ID] = fixed_data

    
    # 3. 「ユーザーの入力」を土台にコピー
    
    # (a) ユーザー(29番)の「行」 (29番 -> 0〜28番への評価)
    # user_evaluations (29要素) を full_matrix[29, 0:29] に設定
    full_matrix[USER_ID, :USER_ID] = user_evaluations
    
    # (b) ユーザー(29番)の「列」 (0〜28番 -> 29番への評価)
    # ルール通り「すべて0で固定」なので、何もしない (土台が0のため)

    return full_matrix