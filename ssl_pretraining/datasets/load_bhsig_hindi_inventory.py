import pandas as pd
from pathlib import Path
from typing import Optional


class InventoryLoader:
    """Trainer class for self-supervised pretraining on signature datasets."""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the trainer and load the inventory CSV.

        Args:
            csv_path: Path to the inventory CSV file. If None, uses default path
                     relative to project root.
        """
        self.csv_path = self._resolve_csv_path(csv_path)
        self.inventory_df = None
        self.load_inventory()

    def _resolve_csv_path(self, csv_path: Optional[str]) -> Path:
        """
        Resolve the CSV path relative to project root if not provided.

        Args:
            csv_path: Explicit path to CSV or None to use default

        Returns:
            Resolved Path object
        """
        if csv_path is None:
            # Resolve relative to project root
            script_dir = Path(__file__).parent.resolve()
            project_root = script_dir.parent.parent
            csv_path = project_root / "artifacts" / "csv" / "ssl_pretraining_inventory.csv"
        else:
            csv_path = Path(csv_path).resolve()

        return csv_path

    def load_inventory(self) -> pd.DataFrame:
        """
        Load the inventory CSV file into a pandas DataFrame.

        Returns:
            DataFrame with columns: writer_id, signature_type, image_path

        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If the DataFrame is empty
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Inventory CSV not found at: {self.csv_path}")

        print(f"Loading inventory from: {self.csv_path}")
        self.inventory_df = pd.read_csv(str(self.csv_path))

        if self.inventory_df.empty:
            raise ValueError(f"Inventory CSV is empty: {self.csv_path}")

        print(f"Inventory loaded successfully with {len(self.inventory_df)} records")
        print(f"\nDataFrame Info:")
        print(f"  Shape: {self.inventory_df.shape}")
        print(f"  Columns: {list(self.inventory_df.columns)}")
        print(f"  Data types:\n{self.inventory_df.dtypes}")

        # Print summary statistics
        print(f"\nSignature Type Distribution:")
        print(self.inventory_df['signature_type'].value_counts())

        print(f"\nWriter Distribution:")
        print(f"  Unique writers: {self.inventory_df['writer_id'].nunique()}")
        print(f"  Signatures per writer (mean): {len(self.inventory_df) / self.inventory_df['writer_id'].nunique():.2f}")

        return self.inventory_df

    def get_inventory(self) -> pd.DataFrame:
        """
        Get the loaded inventory DataFrame.

        Returns:
            DataFrame with signature inventory
        """
        if self.inventory_df is None:
            raise ValueError("Inventory not loaded. Call load_inventory() first.")
        return self.inventory_df

    def get_summary(self) -> dict:
        """
        Get a summary of the inventory.

        Returns:
            Dictionary with inventory statistics
        """
        if self.inventory_df is None:
            raise ValueError("Inventory not loaded. Call load_inventory() first.")

        return {
            'total_records': len(self.inventory_df),
            'unique_writers': self.inventory_df['writer_id'].nunique(),
            'genuine_signatures': (self.inventory_df['signature_type'] == 'G').sum(),
            'forged_signatures': (self.inventory_df['signature_type'] == 'F').sum(),
            'signature_type_distribution': self.inventory_df['signature_type'].value_counts().to_dict(),
        }


if __name__ == "__main__":
    # Test loading the inventory
    trainer = InventoryLoader()
    print("\n" + "="*50)
    print("Inventory Summary:")
    print("="*50)
    summary = trainer.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
