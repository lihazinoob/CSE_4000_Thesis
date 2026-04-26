from pathlib import Path

import torch

from thesis_final_supervised.src.visualization.feature_map_extractor import FeatureMapExtractor
from thesis_final_supervised.src.visualization.feature_map_plotter import (
    plot_feature_grid,
    plot_summary_and_overlay,
)
from thesis_final_supervised.src.utils.create_input_image_tensor import create_signature_transform
from thesis_final_supervised.src.utils.model_loader import load_pretrained_model
from thesis_final_supervised.src.utils.preprocess import preprocess_image
from thesis_final_supervised.src.utils.image_loading import read_image


def main():
    image_path = "data/bhsig-genuine.tif"
    figures_directory = Path("figures")

    # Read the image
    input_image = read_image(
        image_path= image_path,
    )
    # Preprocess the image
    preprocessed_image = preprocess_image(
        image = input_image,
        target_size = (512, 512),
    )


    # Convert the preprocessed_image to a tensor
    image_tensor = create_signature_transform()(
        preprocessed_image
    )


    # Initialize the processing Unit (CPU or GPU )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the path of the encoder model
    encoder_model_path = r"D:\Learning_Education_and_Application\Thesis_results\Self_Supervised_Learning_Result_BHSig260_Bengali_Corrected\saved_models\BHSig260-Bengali_Corrected_R=1_SSL_v5_Encoder_final.pth"

    # Now load the model
    encoder_model = load_pretrained_model(
        device = device,
        model_path = encoder_model_path,
    ).to(device)

    # Set the encoder model to evaluation mode
    encoder_model.eval()

    input_tensor = image_tensor.unsqueeze(0).to(device)
    target_layer = encoder_model.stage4.blocks[1].out_act

    extractor = FeatureMapExtractor(
        model=encoder_model,
        target_layer=target_layer
    )

    first_layer_feature_map = extractor.extract(input_tensor)
    print("Captured feature map shape:", first_layer_feature_map.shape)

    plot_feature_grid(
        feature_map=first_layer_feature_map,
        max_channels=256,
        cols=8,
        cmap="inferno",
        title="Stage 4 Fourth Layer Post-ReLU",
        save_path=figures_directory / "stage_4_fourth_layer_post_relu_feature_maps.png"
    )

    plot_summary_and_overlay(
        input_tensor=input_tensor,
        feature_map=first_layer_feature_map,
        summary_mode="abs_mean",
        heatmap_cmap="inferno",
        title_prefix="Stage 4 Fourth Layer Post-ReLU",
        save_path=figures_directory / "stage_4_fourth_layer_post_relu_summary_overlay.png"
    )

    extractor.close()









if __name__ == "__main__":
    main()
