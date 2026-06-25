from typing import Tuple
import numpy as np
import cv2


def preprocess_image(
        image: np.ndarray,
        target_size: Tuple[int, int]
) -> np.ndarray:
    if image is None:
        raise ValueError('Received a null image during preprocessing.')

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coordinates = cv2.findNonZero(thresh)
    if coordinates is None:
        raise ValueError('No foreground pixels were detected in the image.')

    x, y, w, h = cv2.boundingRect(coordinates)
    crop = image[y:y + h, x:x + w]

    h_c, w_c = crop.shape
    max_dim = max(h_c, w_c)
    square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

    x_offset = (max_dim - w_c) // 2
    y_offset = (max_dim - h_c) // 2
    square_canvas[y_offset:y_offset + h_c, x_offset:x_offset + w_c] = crop

    resized_img = cv2.resize(square_canvas, target_size, interpolation=cv2.INTER_AREA)
    _, final_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return final_img
