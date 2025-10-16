from fastapi import APIRouter
from typing import List
from api.services import timetable
from api.schemas.timetable import Timetable

router = APIRouter()

@router.get("/timetable", response_model=List[Timetable])
async def get_timetable():
    """
    時間割データを全件取得するエンドポイント
    """
    return timetable.get_timetable_all()