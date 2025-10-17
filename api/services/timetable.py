import csv
from typing import List

TIMETABLE_CSV_PATH = "data/timetable.csv"

def get_tametable_by_day(weekday: str) -> List[str]:
    with open(TIMETABLE_CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        timetable_data = list(reader)

    timetable_day = []
    for row in timetable_data:
        if weekday == row[2]:
            timetable_day.append(row[1])

    return timetable_day