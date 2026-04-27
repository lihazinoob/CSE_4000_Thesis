import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history_df: pd.DataFrame, save_path: str = None) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history_df['epoch'], history_df['train_loss'], marker='o', linewidth=2, label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], marker='o', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dual-Triplet Loss')
    plt.title('Train vs Fixed-Validation Loss')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
