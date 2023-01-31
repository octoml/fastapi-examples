import os
import time
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File

from servers.utils.ort import ORTModel
from servers.utils.preprocess import (
    bytes_to_float,
    image_load,
    image_resize_letterboxed,
)

# Load the model
model_path = os.environ.get("MODEL_PATH", "models/yolov5s.onnx")
model_threadcount = int(os.environ.get("MODEL_INTRAOP_THREADS", "0"))
model = ORTModel.load(model_path, intraop_thread_count=model_threadcount)
print(
    f"Loaded model from {model_path}, intraop threads = {model_threadcount}, inputs= {model.inputs}, outputs={model.output_names}"
)

# FastAPI server setup
app = FastAPI()


@app.post("/predict/image")
def predict_image(
    image: bytes = File(...),
):
    model_input_def = model.inputs[0]
    expected_image_w, expected_image_h = (
        model_input_def.shape[2],
        model_input_def.shape[3],
    )

    # Extract and preprocess
    preprocess_start_ns = time.perf_counter_ns()
    # Load image
    source_img = image_load(image)
    # Resize
    target_img, _target_ratio, _target_padding = image_resize_letterboxed(
        source_img, desired_shape=(expected_image_w, expected_image_h)
    )
    # HWC to CHW, BGR to RGB conversion
    target_img = target_img.transpose((2, 0, 1))[::-1]
    target_img = np.expand_dims(target_img, axis=0)

    # target_bytes = target_img.tobytes()
    # with open("zidane.bin", "wb") as target_file:
    #     target_file.write(target_bytes)

    # with open("zidane.bin", "rb") as source_file:
    #     source_bytes = source_file.read()
    #     target_img = np.frombuffer(source_bytes, dtype=np.uint8)
    #     target_img = target_img.reshape(model_input_def.shape)

    # # target_img.tofile("zidane.ndarray")
    # # target_img = np.fromfile("zidane.ndarray", dtype=np.uint8)
    # # target_img = target_img.reshape(model_input_def.shape)

    # Byte to float
    image_input = bytes_to_float(target_img)
    # Insert batch axis
    preprocess_duration_ns = time.perf_counter_ns() - preprocess_start_ns

    # Run inference
    inference_start_ns = time.perf_counter_ns()
    result = model.session.run(model.output_names, {model_input_def.name: image_input})
    inference_duration_ns = time.perf_counter_ns() - inference_start_ns

    result = {
        "input_shape": image_input.shape,
        "output_shape": result[0].shape,
        "preprocess_ms": preprocess_duration_ns / 1e6,
        "inference_ms": inference_duration_ns / 1e6,
    }
    return result


@app.post("/predict/tensor")
def predict_tensor(
    tensor: bytes = File(...),
):
    model_input_def = model.inputs[0]

    # Extract and preprocess
    preprocess_start_ns = time.perf_counter_ns()
    source_tensor = np.frombuffer(tensor, dtype=np.uint8)
    source_tensor = source_tensor.reshape(model_input_def.shape)
    image_input = bytes_to_float(source_tensor)
    preprocess_duration_ns = time.perf_counter_ns() - preprocess_start_ns

    # Run inference
    inference_start_ns = time.perf_counter_ns()
    result = model.session.run(model.output_names, {model_input_def.name: image_input})
    inference_duration_ns = time.perf_counter_ns() - inference_start_ns

    result = {
        "input_shape": image_input.shape,
        "output_shape": result[0].shape,
        "preprocess_ms": preprocess_duration_ns / 1e6,
        "inference_ms": inference_duration_ns / 1e6,
    }
    return result
