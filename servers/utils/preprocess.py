import typing

import cv2
import numpy as np


def image_load(buffer: bytes) -> np.ndarray:
    data = np.frombuffer(buffer, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def image_resize_letterboxed(
    image: np.ndarray,
    desired_shape=(640, 640),
    letterbox_color=(114, 114, 114),
) -> typing.Tuple[np.ndarray, typing.Tuple[float, float], typing.Tuple[float, float]]:

    # Resize and pad image while meeting stride-multiple constraints
    # current shape [height, width]
    shape = image.shape[:2]

    # Scale ratio (new / old)
    ratio = min(desired_shape[0] / shape[0], desired_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    # wh padding
    dw, dh = (
        desired_shape[1] - new_unpad[0],
        desired_shape[0] - new_unpad[1],
    )
    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=letterbox_color
    )
    return image, ratio, (dw, dh)


def bytes_to_float(arr: np.ndarray) -> np.ndarray:
    arr_as_float = arr.astype(np.float32)
    arr_as_float /= 255
    return arr_as_float
