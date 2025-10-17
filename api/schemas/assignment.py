from pydantic import BaseModel

class Assignment(BaseModel):
    id: int
    subject: str
    title: str
    description: str
    due_date: str
    status: str