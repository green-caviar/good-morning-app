from pydantic import BaseModel

class Timetable(BaseModel):
    id: int
    subject: str
    weekday: str
    time: int