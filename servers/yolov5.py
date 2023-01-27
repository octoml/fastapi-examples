import os
import time

import numpy as np
from fastapi import FastAPI, File

from servers.utils.ort import ORTModel
from servers.utils.preprocess import (
    image_load,
    image_resize_letterboxed,
    image_to_float,
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


@app.post("/predict")
async def predict(
    image_file: bytes = File(...),
):
    model_input_def = model.inputs[0]
    expected_image_w, expected_image_h = (
        model_input_def.shape[2],
        model_input_def.shape[3],
    )

    # Extract and preprocess
    preprocess_start_ns = time.perf_counter_ns()
    # Load image
    source_img = image_load(image_file)
    # Resize
    target_img, _target_ratio, _target_padding = image_resize_letterboxed(
        source_img, desired_shape=(expected_image_w, expected_image_h)
    )
    # HWC to CHW, BGR to RGB conversion
    target_img = target_img.transpose((2, 0, 1))[::-1]
    # Byte to float
    target_img = image_to_float(target_img)
    # Insert batch axis
    image_input = np.expand_dims(target_img, axis=0)
    preprocess_duration_ns = time.perf_counter_ns() - preprocess_start_ns

    # Run inference
    inference_start_ns = time.perf_counter_ns()
    result = model.session.run(model.output_names, {model_input_def.name: image_input})
    inference_duration_ns = time.perf_counter_ns() - inference_start_ns

    result = {
        "source_shape": source_img.shape,
        "model_input_shape": image_input.shape,
        "output_shape": result[0].shape,
        "preprocess_time_ms": preprocess_duration_ns / 1e6,
        "inference_time_ms": inference_duration_ns / 1e6,
    }
    return result
