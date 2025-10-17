from fastapi import FastAPI

from api.routers import timetable, assignment

app = FastAPI()
app.include_router(timetable.router)
app.include_router(assignment.router)