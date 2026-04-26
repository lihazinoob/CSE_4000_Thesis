from pathlib import Path

import torch


from thesis_final_supervised.src.utils.create_input_image_tensor import create_signature_transform
from thesis_final_supervised.src.utils.evaluation_data import gather_test_writer_signature_paths, sample_reference_and_query_genuine_paths
from thesis_final_supervised.src.utils.evaluation_embeddings import encode_reference_genuine_paths, \
    build_writer_prototypes
from thesis_final_supervised.src.utils.model_loader import load_verification_model
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
    verification_model_path = r"D:\Learning_Education_and_Application\Thesis_results\Downstream_Verification_Result_BHSig260_Bengali\saved_models\best_triplet_loss_verification_json_writer_disjoint.pt"

    # Now load the model
    verification_model = load_verification_model(
        device = device,
        model_path = verification_model_path,
    ).to(device)



    # Set the encoder model to evaluation mode
    verification_model.eval()

    # Plot the feature maps for the pretrained encoder
    # generate_feature_map_from_pretrained_encoder(
    #     image_tensor=image_tensor,
    #     encoder_model=encoder_model,
    #     device = device,
    #     figures_directory=figures_directory
    # )

    writer_to_signatures = gather_test_writer_signature_paths(
        split_summary_path=r"D:\Learning_Education_and_Application\Thesis\thesis_final_supervised\misc\BHSig260_Bengali_split_summary.json",
        dataset_root=r"D:\Learning_Education_and_Application\Datasets\Signature Dataset\BHSig260-Bengali\BHSig260-Bengali",
    )

    writer_to_protocol = sample_reference_and_query_genuine_paths(
        writer_to_signatures=writer_to_signatures,
        num_reference_genuine=5,
        seed=2026,
    )
    writer_to_reference_embeddings = encode_reference_genuine_paths(
        model=verification_model,
        writer_to_protocol=writer_to_protocol,
        device=device,
        target_size=(512, 512),
    )

    writer_to_prototype = build_writer_prototypes(writer_to_reference_embeddings)





if __name__ == "__main__":
    main()
