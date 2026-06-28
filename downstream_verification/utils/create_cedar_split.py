"""
Creates the CEDAR writer split JSON for the Mendeley-SSL frozen-encoder ablation study.

Partitions all 55 CEDAR writers (1–55) into three non-overlapping sets:
  25 downstream_verification_training
  10 downstream_verification_validation
  20 downstream_verification_testing

There is no SSL pretraining split because the encoder used in this ablation was
pretrained on the Mendeley dataset — CEDAR writers are used exclusively for
downstream evaluation.

Output: artifacts_new/json/cedar_downstream_mendeley_ssl.json
"""

import json
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CEDAR_DATA_DIR = _PROJECT_ROOT / 'data' / 'all' / 'CEDAR'
_OUTPUT_JSON_PATH = _PROJECT_ROOT / 'artifacts_new' / 'json' / 'cedar_downstream_mendeley_ssl.json'

_TRAIN_COUNT = 25
_VAL_COUNT = 10
_TEST_COUNT = 20
_SPLIT_SEED = 42


def create_cedar_split() -> None:
    writer_ids = sorted(
        [d.name for d in _CEDAR_DATA_DIR.iterdir() if d.is_dir()],
        key=lambda x: int(x),
    )

    total = len(writer_ids)
    expected_total = _TRAIN_COUNT + _VAL_COUNT + _TEST_COUNT
    if total != expected_total:
        raise ValueError(
            f'Expected {expected_total} CEDAR writers, found {total}. '
            f'Check that _CEDAR_DATA_DIR is correct: {_CEDAR_DATA_DIR}'
        )

    rng = random.Random(_SPLIT_SEED)
    shuffled = writer_ids.copy()
    rng.shuffle(shuffled)

    train_ids = shuffled[:_TRAIN_COUNT]
    val_ids = shuffled[_TRAIN_COUNT:_TRAIN_COUNT + _VAL_COUNT]
    test_ids = shuffled[_TRAIN_COUNT + _VAL_COUNT:]

    split = {
        'downstream_verification_training': train_ids,
        'downstream_verification_validation': val_ids,
        'downstream_verification_testing': test_ids,
    }

    _OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_JSON_PATH, 'w') as f:
        json.dump(split, f, indent=2)

    print(f'CEDAR writer split saved -> {_OUTPUT_JSON_PATH}')
    print(f'  Training   ({_TRAIN_COUNT} writers): {sorted(train_ids, key=int)}')
    print(f'  Validation ({_VAL_COUNT} writers):  {sorted(val_ids, key=int)}')
    print(f'  Testing    ({_TEST_COUNT} writers):  {sorted(test_ids, key=int)}')


if __name__ == '__main__':
    create_cedar_split()
