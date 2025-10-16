import csv
from typing import List
from api.schemas.timetable import Timetable

TIMETABLE_CSV_PATH = "data/timetable.csv"

def get_timetable_all() -> List[Timetable]:
    """
    CSVファイルから全ての時間割データを読み込み、リストとして返す。
    """
    timetable_list = []
    with open(TIMETABLE_CSV_PATH, mode='r', encoding='utf-8') as f:
        # CSVを辞書形式で読み込む
        reader = csv.DictReader(f)
        for row in reader:
            # Pydanticモデルに変換してリストに追加
            timetable_list.append(Timetable(**row))
    return timetable_list