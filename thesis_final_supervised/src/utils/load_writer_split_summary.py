import json
from typing import Dict


def load_writer_split_summary(
        split_summary_path: str
) -> Dict[str, object]:
    with open(split_summary_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)
