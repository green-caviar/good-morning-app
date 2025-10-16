from fastapi import APIRouter
from typing import List
from api.services import assignment
from api.schemas.assignment import Assignment

router = APIRouter()

@router.get("/assignment", response_model=List[Assignment])
async def get_assignment():
    pass