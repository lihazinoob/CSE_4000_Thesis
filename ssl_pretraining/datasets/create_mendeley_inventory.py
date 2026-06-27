import json
import random
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / 'data' / 'all' / 'Medeley'
_JSON_DIR = _PROJECT_ROOT / 'artifacts' / 'json'
_SPLIT_JSON = _JSON_DIR / 'mendeley_writer_split.json'

NUM_EVAL_WRITERS = 40
RANDOM_SEED = 42


def create_mendeley_inventory(
    data_dir: Path = None,
    output_json_path: Path = None,
    num_eval_writers: int = NUM_EVAL_WRITERS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Scan the Medeley dataset, split 200 writers into train/eval, and save a JSON inventory.

    Folder structure expected:
        data/all/Medeley/
            a_(1)/   1.png ... 30.png
            a_(2)/   ...
            ...
            a_(200)/

    Args:
        data_dir:          Path to the Medeley root folder.
        output_json_path:  Where to write mendeley_writer_split.json.
        num_eval_writers:  Number of writers held out for evaluation (default 40).
        random_seed:       Seed for reproducible writer split (default 42).

    Returns:
        The split dict that was written to JSON.
    """
    data_dir = Path(data_dir) if data_dir else _DATA_DIR
    output_json_path = Path(output_json_path) if output_json_path else _SPLIT_JSON

    if not data_dir.exists():
        raise FileNotFoundError(f'Medeley data directory not found: {data_dir}')

    # Collect all writer folder names (immediate subdirectories only)
    all_writers = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir()
    )

    if len(all_writers) != 200:
        print(f'Warning: expected 200 writer folders, found {len(all_writers)}')

    # Reproducible random split
    rng = random.Random(random_seed)
    shuffled = all_writers.copy()
    rng.shuffle(shuffled)

    eval_writers = sorted(shuffled[:num_eval_writers])
    train_writers = sorted(shuffled[num_eval_writers:])

    # Verify every writer folder contains exactly 30 PNG files
    missing, wrong_count = [], []
    for writer in all_writers:
        writer_dir = data_dir / writer
        pngs = sorted(writer_dir.glob('*.png'))
        if not pngs:
            missing.append(writer)
        elif len(pngs) != 30:
            wrong_count.append((writer, len(pngs)))

    if missing:
        print(f'Warning: {len(missing)} writer folders have no PNG files: {missing}')
    if wrong_count:
        print(f'Warning: {len(wrong_count)} writer folders do not have exactly 30 PNGs:')
        for w, n in wrong_count:
            print(f'  {w}: {n} files')

    split = {
        'dataset': 'Mendeley',
        'data_dir': str(data_dir),
        'total_writers': len(all_writers),
        'num_training_pool': len(train_writers),
        'num_fixed_eval_pool': len(eval_writers),
        'random_seed': random_seed,
        'training_pool': train_writers,
        'fixed_evaluation_pool': eval_writers,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f'Mendeley writer split saved → {output_json_path}')
    print(f'  training_pool        : {len(train_writers)} writers × 30 = {len(train_writers) * 30} images')
    print(f'  fixed_evaluation_pool: {len(eval_writers)} writers × 30 = {len(eval_writers) * 30} images')

    return split


if __name__ == '__main__':
    create_mendeley_inventory()
