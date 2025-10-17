from fastapi import APIRouter
from typing import List
from api.services import timetable
from api.schemas.timetable import Timetable

router = APIRouter()

@router.get("/timetable")
async def get_timetable(weekday: str):
    """
    時間割データを全件取得するエンドポイント
    """
    return timetable.get_tametable_by_day(weekday)