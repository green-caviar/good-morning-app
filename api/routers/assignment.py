from fastapi import APIRouter
from typing import List
from api.services import assignment
from api.schemas.assignment import Assignment

router = APIRouter()

@router.get("/assignment")
async def get_assignment(due_date: str):
    """
    受け取った締切と同じ課題を返す
    """
    return assignment.get_assignment_by_due(due_date)
