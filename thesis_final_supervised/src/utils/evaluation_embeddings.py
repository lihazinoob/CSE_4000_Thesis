from typing import Dict, List

import torch
import torch.nn.functional as F

from thesis_final_supervised.src.utils.create_input_image_tensor import create_signature_transform
from thesis_final_supervised.src.utils.image_loading import read_image
from thesis_final_supervised.src.utils.preprocess import preprocess_image


def create_signature_input_tensor(
        image_path: str,
        target_size=(512, 512)
) -> torch.Tensor:
    input_image = read_image(image_path=image_path)
    if input_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    preprocessed_image = preprocess_image(
        image=input_image,
        target_size=target_size,
    )

    image_tensor = create_signature_transform()(preprocessed_image)
    return image_tensor


@torch.no_grad()
def encode_signature_paths(
        model,
        image_paths: List[str],
        device: torch.device,
        target_size=(512, 512),
) -> Dict[str, torch.Tensor]:
    model.eval()
    path_to_embedding = {}

    for image_path in image_paths:
        image_tensor = create_signature_input_tensor(
            image_path=image_path,
            target_size=target_size,
        ).unsqueeze(0).to(device)

        embedding = model(image_tensor).squeeze(0).detach().cpu()
        path_to_embedding[str(image_path)] = embedding

    return path_to_embedding


@torch.no_grad()
def encode_reference_genuine_paths(
        model,
        writer_to_protocol: Dict[int, Dict[str, List[str]]],
        device: torch.device,
        target_size=(512, 512),
) -> Dict[int, Dict[str, torch.Tensor]]:
    model.eval()
    writer_to_reference_embeddings = {}

    for writer_id in sorted(writer_to_protocol):
        reference_genuine_paths = writer_to_protocol[writer_id]['reference_genuine_paths']
        path_to_embedding = encode_signature_paths(
            model=model,
            image_paths=reference_genuine_paths,
            device=device,
            target_size=target_size,
        )
        writer_to_reference_embeddings[writer_id] = path_to_embedding

    return writer_to_reference_embeddings


def build_normalized_writer_prototype(
        reference_embeddings: List[torch.Tensor]
) -> torch.Tensor:
    if not reference_embeddings:
        raise ValueError("Reference embedding list is empty.")

    prototype_embedding = torch.stack(reference_embeddings, dim=0).mean(dim=0)
    prototype_embedding = F.normalize(
        prototype_embedding.unsqueeze(0),
        p=2,
        dim=1
    ).squeeze(0)

    print("final prototype shape:", prototype_embedding.shape)
    return prototype_embedding


def build_writer_prototypes(
        writer_to_reference_embeddings: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, torch.Tensor]:
    writer_to_prototype = {}

    for writer_id in sorted(writer_to_reference_embeddings):
        reference_embeddings = list(
            writer_to_reference_embeddings[writer_id].values()
        )
        writer_to_prototype[writer_id] = build_normalized_writer_prototype(
            reference_embeddings=reference_embeddings
        )

    return writer_to_prototype
