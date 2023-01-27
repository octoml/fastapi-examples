import typing
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort

_ONNX_TO_NP_TYPE_MAP = {
    "tensor(bool)": np.bool,
    "tensor(int)": np.int32,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(uint64)": np.uint64,
    "tensor(int64)": np.int64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(string)": np.string_,
}


@dataclass(frozen=True)
class ORTInputSpec:
    name: str
    dtype: np.dtype
    shape: typing.List[int]


@dataclass(frozen=True)
class ORTModel:
    session: ort.InferenceSession
    inputs: typing.List[ORTInputSpec]
    output_names: typing.List[str]

    @staticmethod
    def load(
        model_path: str,
        execution_provider: str = "CPUExecutionProvider",
        intraop_thread_count: int = 0,
    ) -> "ORTModel":
        model = onnx.load(model_path)
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if intraop_thread_count:
            session_options.intra_op_num_threads = intraop_thread_count
        session = ort.InferenceSession(
            model.SerializeToString(), session_options, providers=[execution_provider]
        )
        output_names = [x.name for x in session.get_outputs()]
        input_defs = []
        session_inputs = session.get_inputs()
        for input in session_inputs:
            input_name = input.name
            input_type = _ONNX_TO_NP_TYPE_MAP[input.type]
            input_dim = []
            for shape in input.shape:
                # We do not support dynamic axes yet
                assert type(shape) is not str
                input_dim.append(1 if shape is None else shape)
            input_defs.append(
                ORTInputSpec(name=input_name, dtype=input_type, shape=input_dim)
            )
        return ORTModel(session=session, inputs=input_defs, output_names=output_names)
