# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for TensorRT CVNodes."""

from typing import List

import numpy as np
import rclpy
import tensorrt as trt
from cuda import cudart

import cvnode_base.utils.tensorrt_helper as trt_utils
from cvnode_base.core.cvnode_base import BaseCVNode


class CVNodeTensorRTBase(BaseCVNode):
    """
    Base class for TensorRT CVNodes.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name)
        self.declare_parameter("model_path", rclpy.Parameter.Type.STRING)

    def prepare(self):
        engine_path = self.get_parameter("model_path").value
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        # Load TRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = engine.create_execution_context()

        self.stream = trt_utils.cuda_call(cudart.cudaStreamCreate())

        # Allocate memory for inputs and outputs
        # Define input and output specifications
        self.input_specs = []
        self.output_specs = []
        self.bindings = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            shape = engine.get_tensor_shape(name)
            size = trt.volume(shape)
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size

            memory = trt_utils.HostDeviceMemory(size, dtype)
            self.bindings.append((memory.device))
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "size": size,
                "memory": memory,
            }
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_specs.append(binding)
            else:
                self.output_specs.append(binding)
        del engine
        return True

    def predict(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on a batch of preprocessed data.

        Parameters
        ----------
        X : List[np.ndarray]
           Input data.

        Returns
        -------
        List[np.ndarray]
           Inference results.
        """
        for data, input_buffer in zip(X, self.input_specs):
            np.copyto(input_buffer["memory"].host, data.flat)

        for inp in self.input_specs:
            trt_utils.cuda_call(
                cudart.cudaMemcpyAsync(
                    inp["memory"].device,
                    inp["memory"].host,
                    inp["memory"].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

        self.trt_context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream
        )

        for out in self.output_specs:
            trt_utils.cuda_call(
                cudart.cudaMemcpyAsync(
                    out["memory"].host,
                    out["memory"].device,
                    out["memory"].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )

        trt_utils.cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return [
            out["memory"].host.reshape(out["shape"])
            for out in self.output_specs
        ]

    def cleanup(self):
        trt_utils.cuda_call(cudart.cudaStreamDestroy(self.stream))
        for buffer in self.input_specs + self.output_specs:
            buffer["memory"].free()
        del self.trt_context
        del self.stream
        trt_utils.cuda_call(cudart.cudaDeviceReset())
