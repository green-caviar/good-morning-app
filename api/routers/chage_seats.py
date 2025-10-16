from fastapi import APIRouter
from typing import List
from api.services import chage_seats
from api.schemas.change_seats import Change_seats

router = APIRouter()

@router.get("/chage_seats", response_model=List[Change_seats])
async def get_chage_seats():
    pass