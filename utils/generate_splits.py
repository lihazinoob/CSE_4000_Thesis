import json
import random
from pathlib import Path
from typing import Dict, List


def generate_writer_splits(
    data_dir: str = "data/all/BHSig260_Hindi",
    output_dir: str = "artifacts/json",
    output_filename: str = "bh_sig_hindi_writer_split.json",
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Generate random writer splits for different training stages.

    Args:
        data_dir: Path to BHSig260_Hindi directory containing writer folders
        output_dir: Directory where the JSON file will be saved
        output_filename: Name of the output JSON file
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing the splits

    Distribution:
        - self_supervised_pretraining: 80 writers
        - downstream_verification_training: 50 writers
        - downstream_verification_validation: 15 writers
        - downstream_verification_testing: 15 writers
        Total: 160 writers
    """

    # Set seed for reproducibility
    random.seed(seed)

    # Get the absolute path
    data_path = Path(data_dir).resolve()

    # Get all writer folder names
    writer_folders = sorted([
        folder.name for folder in data_path.iterdir()
        if folder.is_dir()
    ])

    print(f"Found {len(writer_folders)} writer folders")

    # Shuffle the writer list
    random.shuffle(writer_folders)

    # Define split sizes
    self_supervised_count = 80
    training_count = 50
    validation_count = 15
    testing_count = 15

    # Split writers
    self_supervised_writers = writer_folders[:self_supervised_count]
    training_writers = writer_folders[self_supervised_count:self_supervised_count + training_count]
    validation_writers = writer_folders[self_supervised_count + training_count:self_supervised_count + training_count + validation_count]
    testing_writers = writer_folders[self_supervised_count + training_count + validation_count:]

    # Create splits dictionary
    splits = {
        "self_supervised_pretraining": self_supervised_writers,
        "downstream_verification_training": training_writers,
        "downstream_verification_validation": validation_writers,
        "downstream_verification_testing": testing_writers
    }

    # Verify totals
    total_writers = sum(len(writers) for writers in splits.values())
    print(f"Total writers in splits: {total_writers}")
    print(f"Distribution:")
    for split_name, writers in splits.items():
        print(f"  {split_name}: {len(writers)}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    json_file_path = output_path / output_filename
    with open(json_file_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplits saved to: {json_file_path}")

    return splits


if __name__ == "__main__":
    # Generate splits when script is run directly
    splits = generate_writer_splits()
