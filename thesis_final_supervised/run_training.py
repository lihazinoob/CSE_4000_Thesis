import torch
from pathlib import Path

from thesis_final_supervised.src.dataset.Dataset_Constructor import create_dataset
from thesis_final_supervised.src.model.DownstreamVerificationNetwork import SignatureVerificationNetwork
from thesis_final_supervised.src.train.loss import DualTripletLoss
from thesis_final_supervised.src.train.trainer import get_optimizer_param_groups, train_model
from thesis_final_supervised.src.train.plotter import plot_training_history

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File Paths
    pretrained_encoder_path = r"D:\Learning_Education_and_Application\Thesis_results\Self_Supervised_Learning_Result_BHSig260_Hindi_Dataset_Corrected\saved_models\BHSig260-Hindi_Corrected_R=1_SSL_v5_Encoder_final.pth"
    dataset_root = r"D:\Learning_Education_and_Application\Datasets\Signature Dataset\BHSig260-Hindi\BHSig260-Hindi"
    split_json_path = r"D:\Learning_Education_and_Application\Thesis\thesis_final_supervised\misc\BHSig260_Hindi_split_summary.json"
    
    output_root = Path(__file__).parent / "Hindi_Downstream_Output"
    best_model_path = output_root / "saved_models" / "best_model.pth"
    history_csv_path = output_root / "logs" / "history.csv"
    plot_path = output_root / "figures" / "loss_plot.png"
    
    # Dataset Parameters
    target_size = (512, 512)
    batch_size = 8
    
    print("Loading datasets...")
    loaders = create_dataset(
        split_summary_path=split_json_path,
        dataset_root=dataset_root,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup Network
    print("Initializing model...")
    model = SignatureVerificationNetwork(
        pretrained_ssl_checkpoint_path=pretrained_encoder_path,
        trainable_encoder_stages=(),  # Only projector head receives gradients as required
        projector_hidden_dim=256,
        embedding_dim=256,
        norm_type='batch'
    ).to(device)

    # Setup Loss & Optimizer
    loss_function = DualTripletLoss(
        intra_margin=0.2,
        inter_margin=0.2,
        inter_loss_weight=1.0,
        distance_p=2.0
    )
    
    learning_rate = 1e-4
    weight_decay = 1e-4
    optimizer_param_groups = get_optimizer_param_groups(model, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(optimizer_param_groups, lr=learning_rate)

    # Training logic
    epochs = 10
    
    print("Starting training...")
    history_df = train_model(
        model=model,
        train_dataset=loaders['train_dataset'],
        train_loader=loaders['train_loader'],
        val_loader=loaders['val_loader'],
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        best_model_path=best_model_path,
        history_csv_path=history_csv_path
    )
    
    # End of run, plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_history(history_df, save_path=str(plot_path))
    
    print("Training finished.")

if __name__ == '__main__':
    main()
