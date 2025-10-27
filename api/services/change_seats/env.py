import numpy as np
import pandas as pd  # 1. pandas をインポート

class Env:

    # <--- 修正点: __init__ が 'relations_matrix' を引数として受け取るように変更
    def __init__(self, relations_matrix: np.ndarray):
        """
        外部から渡された関係性マトリクスを使って環境を初期化する
        """

        # <--- 修正点: CSV読み込みロジック (try...except...) をすべて削除

        # 渡されたNumpy配列をそのままインスタンス変数に設定
        self.relations = relations_matrix 

        self.rows = 5
        self.cols = 6
        self.num_students = self.rows * self.cols # 30人

        # マトリクスの形状チェック (これは残します)
        if self.relations.shape != (self.num_students, self.num_students):
            raise ValueError(
                f"関係性マトリクスの形状が({self.num_students}, {self.num_students})ではありません。"
                f"受け取ったShape: {self.relations.shape}"
            )

    def reset(self):
        """席配列を初期化する"""

        initial_state_1d = np.random.permutation(self.num_students)
        initial_state_2d = initial_state_1d.reshape((self.rows, self.cols))

        return initial_state_2d

    def calculate_score(self, state):
        total_score = 0
        
        # --- 定義された重み ---
        WEIGHT_ADJACENT = 1.0   # 前後左右（通路なし）の重み
        WEIGHT_AISLE = 0.5      # 通路を挟む左右の重み
        WEIGHT_DIAGONAL = 0.5   # 斜めの重み (ご要望通り)
        
        # 8方向の「移動先（オフセット）」を定義
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 前後左右
            (-1, 1), (-1, -1), (1, 1), (1, -1) # 斜め4方向
        ]
        
        # 全ての席を2重ループでチェック
        for r in range(self.rows): 
            for c in range(self.cols): 
                
                student1_id = state[r][c] 
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc # nr = new_row, nc = new_col
                    
                    # 境界条件チェック
                    if (0 <= nr < self.rows) and (0 <= nc < self.cols):
                        student2_id = state[nr][nc]
                        
                        # 1. 生徒間の関係性スコア (A->B + B->A)
                        score_raw = (self.relations[student1_id][student2_id] +
                                     self.relations[student2_id][student1_id])

                        # --- 2. 重み付けロジックの開始 ---
                        weight = WEIGHT_ADJACENT # 前後 (dr!=0, dc=0) や デフォルトは 1.0

                        if dc != 0 and dr == 0:
                            # ① 左右の席の場合 (真横)
                            
                            is_across_aisle = False
                            
                            # 通路 1: c=1 と c=2 の間 (c=1の右隣 または c=2の左隣)
                            if (c == 1 and dc == 1) or (c == 2 and dc == -1):
                                is_across_aisle = True
                            
                            # 通路 2: c=3 と c=4 の間 (c=3の右隣 または c=4の左隣)
                            elif (c == 3 and dc == 1) or (c == 4 and dc == -1):
                                is_across_aisle = True

                            if is_across_aisle:
                                weight = WEIGHT_AISLE  # 0.5 (通路を挟む)
                            else:
                                # 通路を挟まない隣 (例: c=0とc=1の間、c=2とc=3の間)
                                weight = WEIGHT_ADJACENT # 1.0 
                                
                        elif dc != 0 and dr != 0:
                            # ② 斜めの席の場合
                            weight = WEIGHT_DIAGONAL # 0.5
                        
                        # 3. 重みを適用してスコアを加算
                        total_score += score_raw * weight
        
        # (A,B)のスコアと(B,A)のスコアを2回足しているため、2で割る
        return total_score / 2

    def step(self, state, action_pair):
        """席を変え、新しい配置と報酬を返す"""

        student_id_1, student_id_2 = action_pair
        try:
            coords1 = np.where(state == student_id_1)
            r1, c1 = coords1[0][0], coords1[1][0]

            coords2 = np.where(state == student_id_2)
            r2, c2 = coords2[0][0], coords2[1][0]
        except IndexError:
            print(f"エラー: 生徒ID {student_id_1} または {student_id_2} が座席表で見つかりません。")
            return state, 0.0, False

        before_score = self.calculate_score(state)

        next_state = state.copy()
        next_state[r1][c1], next_state[r2][c2] = next_state[r2][c2], next_state[r1][c1]

        after_score = self.calculate_score(next_state)

        reward = after_score - before_score

        done = False

        return next_state, reward, done