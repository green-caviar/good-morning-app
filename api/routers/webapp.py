# api/routers/webapp.py

import os
from typing import List, Optional
# ★ Request, Form, Query をインポート (Queryは無くてもよいが念の為)
from fastapi import APIRouter, Request, Form, Query 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# ★ --- ここから追加 --- ★
import httpx # API呼び出し用
import pandas as pd # 名簿(CSV)読み込み用
from starlette.datastructures import URL # ★ これを追加 (または str() で囲む)
try:
    # ROSTER_PATH を読み込むために config をインポート
    from api.services.change_seats import config 
except ImportError as e:
    print(f"CRITICAL: config.py が見つかりません: {e}")
    config = None # 起動失敗の可能性
# ★ --- 追加ここまで --- ★


# 1. あなたのサービスモジュールをインポートします
try:
    from api.services import assignment, timetable
except ImportError:
    print("CRITICAL: 'assignment' または 'timetable' サービスが見つかりません。")
    raise

# 2. ルーターの初期化
router = APIRouter()

# 3. Jinja2テンプレート（お皿）の設定
templates = Jinja2Templates(directory="templates")


# 4. ページを「最初に見る」ときの処理 (GETリクエスト)
@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    """
    トップページ (/) にGETリクエストが来た時 (＝最初にアクセスした時) の処理。
    空のフォームがあるHTMLを返す。
    """
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "assignments_list": [], # 最初は空
        "timetable_list": []    # 最初は空
    })

# 5. フォームを「送信」したときの処理 (POSTリクエスト)
@router.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    due_date: Optional[str] = Form(None), # フォームから "due_date" を受け取る
    weekday: Optional[str] = Form(None)   # フォームから "weekday" を受け取る
):
    """
    トップページ (/) にPOSTリクエストが来た時 (＝フォームが送信された時) の処理。
    データを検索し、結果をHTMLに埋め込んで返す。
    """
    
    # 6. インポートしたサービス関数を呼び出してデータを検索
    # (※ `assignment.py` 側で日付形式の変換 '- '-> '/' が行われる前提)
    assignments_data = assignment.get_assignment_by_due(due_date)
    timetable_data = timetable.get_tametable_by_day(weekday)

    # 7. 検索結果をHTMLに渡して返す
    return templates.TemplateResponse("index.html", {
        "request": request,
        "assignments_list": assignments_data, # 検索結果
        "timetable_list": timetable_data    # 検索結果
    })
    
#
# ... (既存の @router.get("/") と @router.post("/") はこの上にある) ...
#

# --- ▼▼▼ これ以降のコードをファイルの一番下に追加 ▼▼▼ ---

# === 席替え(カスタム)入力ページの表示 (GET) ===
@router.get("/seating", response_class=HTMLResponse)
async def show_seating_form(request: Request):
    """
    好感度入力フォームがある "seating.html" ページを表示する
    ★ 名簿(ROSTER_PATH)を読み込み、生徒名リストをHTMLに渡す
    """
    student_names = []
    user_name = "あなた" # デフォルト
    error_message = None

    try:
        if config is None or config.ROSTER_PATH is None:
            raise FileNotFoundError("config.py または ROSTER_PATH が設定されていません。")
        
        # 1. 名簿を読み込む
        df_roster = pd.read_csv(config.ROSTER_PATH)
        
        # 2. 生徒29人 (USER_ID(29)以外) の名前リストを取得
        student_names = df_roster[
            df_roster['id'] != config.USER_ID
        ]['name'].tolist()
        
        # 3. あなた (USER_ID(29)) の名前を取得
        user_name = df_roster[
            df_roster['id'] == config.USER_ID
        ]['name'].iloc[0]

        if len(student_names) != (config.NUM_STUDENTS - 1):
             error_message = f"名簿の人数が {config.NUM_STUDENTS - 1} 人ではありません。"
             
    except Exception as e:
        error_message = f"名簿ファイル '{config.ROSTER_PATH}' の読み込みに失敗: {e}"

    # 4. seating.html を表示
    return templates.TemplateResponse("seating.html", {
        "request": request,
        "layout": None,
        "student_names": student_names, # ★ 生徒名リスト
        "user_name": user_name,         # ★ あなたの名前
        "error_message": error_message
    })

# === 席替え(カスタム)実行 (POST) ===
@router.post("/seating", response_class=HTMLResponse)
async def process_seating_form(request: Request):
    """
    フォーム(名前:スコア)を受け取り、
    バックエンドAPI (change_seats.py の /recommend) を呼び出し、
    結果を "seating.html" に描画する
    """
    
    # 1. フォームから「名前: スコア」の辞書を作成
    user_evals_dict = {}
    student_names_from_form = [] # エラー時にフォームを復元するため
    user_name_from_form = "あなた" # エラー時にフォームを復元するため
    
    try:
        form_data = await request.form()
        for name, score_str in form_data.items():
            try:
                # スコアをfloatに変換
                score_float = float(score_str)
                user_evals_dict[name] = score_float
            except ValueError:
                # 'student_names' や 'user_name' のような文字列は無視
                if name == "student_names_json":
                    import json
                    student_names_from_form = json.loads(score_str)
                elif name == "user_name":
                    user_name_from_form = score_str
                pass 

    except Exception as e:
        return templates.TemplateResponse("seating.html", {
            "request": request, "layout": None,
            "student_names": student_names_from_form,
            "user_name": user_name_from_form,
            "error_message": f"フォームデータの読み取りエラー: {e}"
        })

    # 2. バックエンドAPIに送るJSONペイロードを作成
    payload = {"user_evals": user_evals_dict}

    # 3. httpx を使ってバックエンドAPI (/recommend) を呼び出す
    layout_result = None
    final_score = None # ★ スコアを初期化
    error_msg = None
    try:
# ▼▼▼ ここの1行を修正 ▼▼▼
        # 修正前: api_url = request.url_for('recommend_seat_change')
        api_url: str = str(request.url_for('recommend_seat_change'))
        # ▲▲▲ str() で囲んで文字列に変換 ▲▲▲
        
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=30.0)
            
            if response.status_code == 200:
                result_data = response.json()
                layout_result = result_data.get("final_layout_names") # 名前入りの座席表
                final_score = result_data.get("final_score") # ★ APIからスコアを取得
                if layout_result is None:
                    error_msg = f"APIは成功しましたが、'final_layout_names' が見つかりません。"
            else:
                error_msg = f"APIエラー (Status {response.status_code}): {response.text}"
                
    except httpx.ConnectError as e:
        error_msg = f"APIへの接続エラー: {e}"
    except httpx.ReadTimeout:
        error_msg = "APIの処理がタイムアウトしました (30秒)。"
    except Exception as e:
        error_msg = f"API呼び出し中に予期せぬエラー: {e}"

    # 4. 結果を描画用HTMLに渡して返す
    return templates.TemplateResponse("seating.html", {
        "request": request,
        "layout": layout_result, # ★ 名前入りの座席表
        "error_message": error_msg,
        "final_score": final_score, # ★ スコアをHTMLに渡す
        "student_names": student_names_from_form, # ★ フォーム復元用
        "user_name": user_name_from_form          # ★ フォーム復元用
    })