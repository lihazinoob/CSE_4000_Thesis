from typing import Tuple

import cv2
import numpy as np
def preprocess_image(
        image : np.ndarray,
        target_size : Tuple[int, int] = (512,512),
) -> np.ndarray:

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(thresh)
    if coordinates is None:
        raise ValueError("No foreground pixels were detected.")

    x, y, width, height = cv2.boundingRect(coordinates)
    crop = image[y:y + height, x:x + width]

    crop_height, crop_width = crop.shape
    max_dim = max(crop_height, crop_width)
    square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

    x_offset = (max_dim - crop_width) // 2
    y_offset = (max_dim - crop_height) // 2
    square_canvas[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = crop

    resized_image = cv2.resize(square_canvas, target_size, interpolation=cv2.INTER_AREA)
    _, final_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return final_image