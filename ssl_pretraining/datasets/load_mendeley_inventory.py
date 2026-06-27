import json
from pathlib import Path
from typing import Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_JSON   = _PROJECT_ROOT / 'artifacts' / 'json' / 'mendeley_writer_split.json'
_CSV_DIR      = _PROJECT_ROOT / 'artifacts' / 'csv'


class MendeleyInventoryLoader:
    """
    Loads the Mendeley writer split JSON and builds train / eval inventory DataFrames.

    Each row in the returned DataFrames has the columns expected by SignatureSSLDataset:
        image_path     – absolute path to the PNG file
        writer_id      – writer folder name, e.g. 'a_(1)'
        signature_type – 'unlabeled' for all Mendeley images

    CSV side-effect
    ---------------
    load_inventory() saves two CSV files to artifacts/csv/:
        mendeley_train_inventory.csv  – 160 training writers  (4 800 rows)
        mendeley_eval_inventory.csv   – 40 held-out writers   (1 200 rows)

    These CSVs are the persistent record of what the model was trained and evaluated
    on.  They are written once at the start of each training run and never modified
    during training.
    """

    def __init__(
        self,
        json_path: str = None,
        data_dir: str = None,
    ):
        self.json_path = Path(json_path) if json_path else _SPLIT_JSON
        # data_dir can override the path stored inside the JSON
        self._data_dir_override = Path(data_dir) if data_dir else None

    # ------------------------------------------------------------------
    def load_inventory(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            train_df – DataFrame for the 160 training writers
            eval_df  – DataFrame for the 40 held-out evaluation writers

        Also saves mendeley_train_inventory.csv and mendeley_eval_inventory.csv
        to artifacts/csv/ as a persistent record of the data split used.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(
                f'Mendeley split JSON not found: {self.json_path}\n'
                f'Run ssl_pretraining/datasets/create_mendeley_inventory.py first.'
            )

        with open(self.json_path, 'r') as f:
            split = json.load(f)

        data_dir = self._data_dir_override or Path(split['data_dir'])

        if not data_dir.exists():
            raise FileNotFoundError(f'Medeley data directory not found: {data_dir}')

        train_df = self._build_dataframe(split['training_pool'], data_dir)
        eval_df  = self._build_dataframe(split['fixed_evaluation_pool'], data_dir)

        self._save_csvs(train_df, eval_df)
        self._print_summary(split, train_df, eval_df)
        return train_df, eval_df

    # ------------------------------------------------------------------
    @staticmethod
    def _build_dataframe(writers: list, data_dir: Path) -> pd.DataFrame:
        rows = []
        for writer in writers:
            writer_dir = data_dir / writer
            if not writer_dir.exists():
                print(f'Warning: writer folder not found, skipping: {writer_dir}')
                continue
            for png in sorted(writer_dir.glob('*.png')):
                rows.append({
                    'image_path':     str(png),
                    'writer_id':      writer,
                    'signature_type': 'unlabeled',
                })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f'No images found for the provided writer list in {data_dir}')
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _save_csvs(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
        _CSV_DIR.mkdir(parents=True, exist_ok=True)
        train_path = _CSV_DIR / 'mendeley_train_inventory.csv'
        eval_path  = _CSV_DIR / 'mendeley_eval_inventory.csv'
        train_df.to_csv(train_path, index=False)
        eval_df.to_csv(eval_path,  index=False)
        print(f'  Saved → {train_path}  ({len(train_df)} rows)')
        print(f'  Saved → {eval_path}   ({len(eval_df)} rows)')

    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(split: dict, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
        print('=' * 50)
        print('Mendeley Inventory Loaded')
        print('=' * 50)
        print(f"  training_pool        : {split['num_training_pool']} writers  |  {len(train_df)} images")
        print(f"  fixed_evaluation_pool: {split['num_fixed_eval_pool']} writers  |  {len(eval_df)} images")
        print(f"  Random seed          : {split['random_seed']}")
        print('=' * 50)


if __name__ == '__main__':
    loader = MendeleyInventoryLoader()
    train_df, eval_df = loader.load_inventory()
    print(train_df.head())
    print(eval_df.head())
