from fastapi import FastAPI

from api.routers import timetable

app = FastAPI()
app.include_router(timetable.router)
