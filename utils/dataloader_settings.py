import os
from typing import Tuple

import torch
def resolve_dataloader_settings(number_of_gpus) -> Tuple[int, bool]:
    if os.name == 'nt':
        num_workers = 0
        pin_memory = True
    else:
        num_workers = min(8, max(2, number_of_gpus * 2)) if torch.cuda.is_available() else 2
        pin_memory = torch.cuda.is_available()
    return num_workers, pin_memory
