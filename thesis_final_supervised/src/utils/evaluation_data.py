import json
import random
from pathlib import Path
from typing import Dict, List


def load_split_summary(split_summary_path: str) -> Dict[str, object]:
    with open(split_summary_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_split_writer_ids(
        split_summary_path: str,
        split_key: str = 'test_writers'
) -> List[int]:
    split_summary = load_split_summary(split_summary_path)
    if split_key not in split_summary:
        raise KeyError(f"Missing split key: {split_key}")
    return [int(writer_id) for writer_id in split_summary[split_key]]


def gather_writer_signature_paths(
        dataset_root: str,
        writer_ids: List[int]
) -> Dict[int, Dict[str, List[str]]]:
    dataset_root_path = Path(dataset_root)
    writer_to_signatures = {}

    for writer_id in sorted(int(writer_id) for writer_id in writer_ids):
        writer_directory = dataset_root_path / str(writer_id)
        if not writer_directory.is_dir():
            raise FileNotFoundError(f"Writer directory not found: {writer_directory}")

        genuine_paths = []
        forgery_paths = []

        for image_path in sorted(writer_directory.iterdir(), key=lambda path: path.name):
            if not image_path.is_file():
                continue

            file_name = image_path.name.upper()
            if '-G-' in file_name:
                genuine_paths.append(str(image_path))
            elif '-F-' in file_name:
                forgery_paths.append(str(image_path))
            else:
                raise ValueError(f"Could not infer signature type from filename: {image_path.name}")

        if not genuine_paths:
            raise ValueError(f"No genuine signatures found for writer {writer_id}")
        if not forgery_paths:
            raise ValueError(f"No forgery signatures found for writer {writer_id}")

        writer_to_signatures[writer_id] = {
            'genuine_paths': genuine_paths,
            'forgery_paths': forgery_paths,
        }

    return writer_to_signatures


def gather_test_writer_signature_paths(
        split_summary_path: str,
        dataset_root: str
) -> Dict[int, Dict[str, List[str]]]:
    return gather_split_writer_signature_paths(
        split_summary_path=split_summary_path,
        dataset_root=dataset_root,
        split_key='test_writers'
    )


def gather_split_writer_signature_paths(
        split_summary_path: str,
        dataset_root: str,
        split_key: str
) -> Dict[int, Dict[str, List[str]]]:
    split_writer_ids = get_split_writer_ids(
        split_summary_path=split_summary_path,
        split_key=split_key
    )
    return gather_writer_signature_paths(
        dataset_root=dataset_root,
        writer_ids=split_writer_ids
    )


def sample_reference_and_query_genuine_paths(
        writer_to_signatures: Dict[int, Dict[str, List[str]]],
        num_reference_genuine: int = 5,
        num_forgery_queries: int = None,
        seed: int = 2026
) -> Dict[int, Dict[str, List[str]]]:
    rng = random.Random(seed)
    writer_to_protocol = {}

    for writer_id in sorted(writer_to_signatures):
        genuine_paths = list(writer_to_signatures[writer_id]['genuine_paths'])
        forgery_paths = list(writer_to_signatures[writer_id]['forgery_paths'])

        if len(genuine_paths) <= num_reference_genuine:
            raise ValueError(
                f"Writer {writer_id} does not have enough genuine samples for "
                f"{num_reference_genuine}-shot evaluation."
            )

        reference_genuine_paths = sorted(
            rng.sample(genuine_paths, num_reference_genuine)
        )
        reference_path_set = set(reference_genuine_paths)
        genuine_query_paths = [
            image_path for image_path in genuine_paths
            if image_path not in reference_path_set
        ]
        if num_forgery_queries is None:
            selected_forgery_query_paths = forgery_paths
        else:
            if len(forgery_paths) < num_forgery_queries:
                raise ValueError(
                    f"Writer {writer_id} does not have enough forgery samples for "
                    f"{num_forgery_queries}-query evaluation."
                )
            selected_forgery_query_paths = sorted(
                rng.sample(forgery_paths, num_forgery_queries)
            )

        writer_to_protocol[writer_id] = {
            'reference_genuine_paths': reference_genuine_paths,
            'genuine_query_paths': genuine_query_paths,
            'forgery_query_paths': selected_forgery_query_paths,
        }

    return writer_to_protocol
