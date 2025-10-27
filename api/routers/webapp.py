import os
from typing import List, Optional
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# 1. あなたのサービスモジュールをインポートします
try:
    from api.services import assignment, timetable
except ImportError:
    # サービスが見つからない場合のエラーハンドリング (仮)
    print("CRITICAL: 'api.services.assignment' または 'api.services.timetable' が見つかりません。")
    # ... (この部分は適宜調整してください) ...
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