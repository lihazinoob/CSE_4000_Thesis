import json
import pandas as pd
from pathlib import Path
from typing import List


def create_bhsig_hindi_inventory(
    split_json_path: str = None,
    data_dir: str = None,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Create an inventory dataframe for self-supervised pretraining writers.

    Reads the writer splits JSON and creates a dataframe containing all signature
    files for writers in the self_supervised_pretraining split.

    Args:
        split_json_path: Path to the bh_sig_hindi_writer_split.json file (default: resolved relative to script)
        data_dir: Path to BHSig260_Hindi directory (default: resolved relative to script)
        output_csv_path: Path where the CSV inventory will be saved (default: resolved relative to script)

    Returns:
        DataFrame with columns: writer_id, signature_type, image_path
    """

    # Resolve paths relative to script location if not provided
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent

    if split_json_path is None:
        split_json_path = project_root / "artifacts" / "json" / "bh_sig_hindi_writer_split.json"
    else:
        split_json_path = Path(split_json_path).resolve()

    if data_dir is None:
        data_dir = project_root / "data" / "all" / "BHSig260_Hindi"
    else:
        data_dir = Path(data_dir).resolve()

    if output_csv_path is None:
        output_csv_path = project_root / "artifacts" / "csv" / "ssl_pretraining_inventory.csv"
    else:
        output_csv_path = Path(output_csv_path).resolve()

    # Load the writer splits
    with open(str(split_json_path), 'r') as f:
        splits = json.load(f)

    # Get writers for self-supervised pretraining
    ssl_writers = splits['self_supervised_pretraining']
    print(f"Processing {len(ssl_writers)} writers for self-supervised pretraining")

    rows = []
    data_path = Path(data_dir).resolve() if not isinstance(data_dir, Path) else data_dir

    # Process each writer
    for writer_id in ssl_writers:
        writer_dir = data_path / writer_id

        if not writer_dir.exists():
            print(f"Warning: Writer directory not found: {writer_dir}")
            continue

        # Get all .tif files in the writer directory
        signature_files = sorted(writer_dir.glob("*.tif"))

        for image_file in signature_files:
            # Extract signature type from filename
            # Filename format: H-S-{writer_id}-{signature_type}-{number}.tif
            parts = image_file.stem.split('-')
            if len(parts) >= 4:
                signature_type = parts[3]  # 'G' for genuine, 'F' for forged
            else:
                print(f"Warning: Could not parse signature type from {image_file.name}")
                continue

            rows.append({
                'writer_id': writer_id,
                'signature_type': signature_type,
                'image_path': str(image_file),
            })

    # Create dataframe
    inventory_df = pd.DataFrame(rows)

    if inventory_df.empty:
        raise ValueError(f'No signature images were found for SSL writers in: {data_dir}')

    print(f"\nInventory created with {len(inventory_df)} signature files")

    # Count statistics
    genuine_count = (inventory_df['signature_type'] == 'G').sum()
    forged_count = (inventory_df['signature_type'] == 'F').sum()

    print(f"Genuine signatures (G): {genuine_count}")
    print(f"Forged signatures (F): {forged_count}")
    print(f"Total signatures: {len(inventory_df)}")

    # Create output directory if it doesn't exist
    if not isinstance(output_csv_path, Path):
        output_path = Path(output_csv_path).resolve()
    else:
        output_path = output_csv_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    inventory_df.to_csv(str(output_path), index=False)
    print(f"\nInventory saved to: {output_path}")

    return inventory_df


if __name__ == "__main__":
    # Create SSL inventory when script is run directly
    inventory_df = create_bhsig_hindi_inventory()
