import csv
from typing import List

ASSIGNMENT_DATA_PATH = "data/assignment.csv"
def get_assignment_by_due(due_date: str) -> List[str]:
    with open(ASSIGNMENT_DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        assignment_data = list(reader)

    assignment = []
    for row in assignment_data:
        if due_date == row[4]:
            assignment.append(row)
    return assignment
#金野泰樹がこのプロジェクトに参加しました。ここの機能を追加します。