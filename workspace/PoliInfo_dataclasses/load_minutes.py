
import json
from pathlib import Path
from .minutes import MinutesObject


def load_minutes(monutes_data_path: str) -> MinutesObject:
    """
    指定したパスから予算項目データを読み込み、MinutesObjectインスタンスとして返す。
    """
    p = Path(monutes_data_path)
    return MinutesObject.from_dict(json.loads(p.read_text(encoding="utf-8")))