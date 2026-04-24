from typing import Any, Optional, Union

import numpy as np
import cv2
from numpy import ndarray


def read_image(
        image_path:str
) -> Union[Optional[ndarray], Any]:
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return input_image