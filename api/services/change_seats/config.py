# api/services/change_seats/config.py

import numpy as np

# --- クラスの基本設定 ---
NUM_STUDENTS = 30
ROWS = 5
COLS = 6

# --- あなた（ユーザー）の設定 ---
# あなたを「出席番号0番」として固定します
USER_ID = 29

# --- 固定データの定義 ---

# ユーザー(29番)以外 (0番〜28番) の生徒同士の関係性を定義します。
# (NUM_STUDENTS - 1) x (NUM_STUDENTS - 1) = 29x29 のマトリクス
#
# ★★★重要★★★
# まずはテストのため、「全員が普通(0)」のダミーデータを使います。
# あとで、この部分を本物の関係性データに置き換えることができます。
CSV_PATH = "data/relationships.csv"
ROSTER_PATH = "data/roster.csv"

# (例: もし1番と2番だけが仲良し(+1)なら、以下のように設定します)
# FIXED_RELATIONS_OTHERS[1-1, 2-1] = 1.0 # 1番 -> 2番
# FIXED_RELATIONS_OTHERS[2-1, 1-1] = 1.0 # 2番 -> 1番