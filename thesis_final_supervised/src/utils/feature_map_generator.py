from pathlib import Path

import torch

from thesis_final_supervised.src.visualization.feature_map_extractor import FeatureMapExtractor
from thesis_final_supervised.src.visualization.feature_map_plotter import plot_feature_grid, plot_summary_and_overlay


def generate_feature_map_from_pretrained_encoder(
        image_tensor : torch.Tensor,
        encoder_model,
        device : torch.device,
        figures_directory : Path,
):
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
