from fastapi import FastAPI

from api.routers import timetable, assignment, change_seats

app = FastAPI()
app.include_router(timetable.router)
app.include_router(assignment.router)
app.include_router(change_seats.router)