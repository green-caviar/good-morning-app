from pydantic import BaseModel

class Timetable(BaseModel):
    period: int
    monday: str
    tuesday: str
    wednesday: str
    thursday: str
    friday: str