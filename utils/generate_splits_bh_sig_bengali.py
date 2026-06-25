import json
import random
from pathlib import Path
from typing import Dict, List


def generate_writer_splits(
    data_dir: str = "data/all/BHSig260_Bengali",
    output_dir: str = "artifacts/json",
    output_filename: str = "bh_sig_bengali_writer_split.json",
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate random writer splits for BHSig260 Bengali dataset.

    Bengali has 100 writers. Distribution:
        - self_supervised_pretraining:      20 writers  (unused in cross-dataset scenario
                                                         but kept for schema consistency)
        - downstream_verification_training: 50 writers
        - downstream_verification_validation: 15 writers
        - downstream_verification_testing:  15 writers
        Total: 100 writers

    Args:
        data_dir:        Path to BHSig260_Bengali directory containing writer folders.
        output_dir:      Directory where the JSON file will be saved.
        output_filename: Name of the output JSON file.
        seed:            Random seed for reproducibility.

    Returns:
        Dictionary containing the writer splits.
    """
    random.seed(seed)

    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    writer_folders = sorted([
        folder.name for folder in data_path.iterdir()
        if folder.is_dir()
    ])

    total = len(writer_folders)
    print(f"Found {total} writer folders in {data_path}")

    ssl_count        = 20
    training_count   = 50
    validation_count = 15
    testing_count    = 15

    expected = ssl_count + training_count + validation_count + testing_count
    if total != expected:
        raise ValueError(
            f"Expected {expected} writers for this split configuration, "
            f"but found {total}."
        )

    random.shuffle(writer_folders)

    idx = 0
    ssl_writers        = writer_folders[idx : idx + ssl_count];        idx += ssl_count
    training_writers   = writer_folders[idx : idx + training_count];   idx += training_count
    validation_writers = writer_folders[idx : idx + validation_count]; idx += validation_count
    testing_writers    = writer_folders[idx : idx + testing_count]

    splits: Dict[str, List[str]] = {
        "self_supervised_pretraining":       ssl_writers,
        "downstream_verification_training":  training_writers,
        "downstream_verification_validation": validation_writers,
        "downstream_verification_testing":   testing_writers,
    }

    print("Distribution:")
    for split_name, writers in splits.items():
        print(f"  {split_name}: {len(writers)} writers")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    json_file_path = output_path / output_filename
    with open(json_file_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplits saved to: {json_file_path}")
    return splits


if __name__ == "__main__":
    splits = generate_writer_splits()
