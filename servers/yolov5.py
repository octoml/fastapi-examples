import os
import time
import typing

import numpy as np
from fastapi import FastAPI, File

from servers.utils.cuda import CudaMem
from servers.utils.ort import ORTModel
from servers.utils.preprocess import (
    bytes_to_float,
    image_load,
    image_resize_letterboxed,
)

# FastAPI server setup
app = FastAPI()
app_model: typing.Optional[ORTModel] = None


def get_model_config():
    model_path = os.environ.get("MODEL_PATH", "models/yolov5s.onnx")
    model_ep = os.environ.get("MODEL_EXECUTION_PROVIDER", "CPUExecutionProvider")
    model_threadcount = int(os.environ.get("MODEL_INTRAOP_THREADS", "0"))
    return model_path, model_ep, model_threadcount


def load_model(
    model_path: str,
    execution_provider: str,
    thread_count: int,
):
    model = ORTModel.load(
        model_path,
        execution_provider=execution_provider,
        intraop_thread_count=thread_count,
    )
    print(
        f"Loaded model: {model_path}, execution provider: {execution_provider}, intraop threads: {thread_count}"
    )
    print(f" - Inputs: {model.inputs}")
    print(f" - Outputs: {model.output_names}")
    return model


@app.on_event("startup")
def startup_event():
    global app_model
    model_path, model_ep, model_threadcount = get_model_config()
    app_model = load_model(model_path, model_ep, model_threadcount)


@app.post("/predict/image")
def predict_image(
    image: bytes = File(...),
):
    global app_model
    model = app_model
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

    # Save to Numpy Array
    # target_img.tofile("zidane.ndarray")
    # target_img = np.fromfile("zidane.ndarray", dtype=np.uint8)
    # target_img = target_img.reshape(model_input_def.shape)

    # Byte to float
    image_input = bytes_to_float(target_img)
    # Insert batch axis
    preprocess_duration_ns = time.perf_counter_ns() - preprocess_start_ns

    # Run inference
    inference_start_ns = time.perf_counter_ns()
    result = model.forward({model_input_def.name: image_input})
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
    global app_model
    model = app_model
    model_input_def = model.inputs[0]

    # Extract and preprocess
    preprocess_start_ns = time.perf_counter_ns()
    source_tensor = np.frombuffer(tensor, dtype=np.uint8)
    source_tensor = source_tensor.reshape(model_input_def.shape)
    image_input = bytes_to_float(source_tensor)
    preprocess_duration_ns = time.perf_counter_ns() - preprocess_start_ns

    # Run inference
    inference_start_ns = time.perf_counter_ns()
    result = model.forward({model_input_def.name: image_input})
    inference_duration_ns = time.perf_counter_ns() - inference_start_ns

    result = {
        "input_shape": image_input.shape,
        "output_shape": result[0].shape,
        "preprocess_ms": preprocess_duration_ns / 1e6,
        "inference_ms": inference_duration_ns / 1e6,
    }
    return result


if __name__ == "__main__":

    # Load model configuration
    model_path, model_ep, model_threadcount = get_model_config()

    # Measure memory if using GPU EPs
    gpu_enabled = (
        model_ep == "CUDAExecutionProvider" or model_ep == "TensorrtExecutionProvider"
    )
    if gpu_enabled:
        cuda_mem_measurer = CudaMem()
        cuda_mem_initial, cuda_mem_total = cuda_mem_measurer.measure()
        print(
            f"CUDA Memory: Total: {cuda_mem_total/(1024*1024):.2f} MB, Free: {cuda_mem_initial/(1024*1024):.2f} MB"
        )

    model = load_model(model_path, model_ep, model_threadcount)

    if gpu_enabled:

        import gc

        inputs = model.random_inputs()
        model.forward(inputs)
        model.forward(inputs)

        gc.collect()

        cuda_mem_current, cuda_mem_total = cuda_mem_measurer.measure()
        cuda_mem_used = cuda_mem_initial - cuda_mem_current

        print(
            f"CUDA Memory: Used: {cuda_mem_used} bytes, {cuda_mem_used/(1024*1024):.2f} MB, Free: {cuda_mem_current/(1024*1024):.2f} MB"
        )
