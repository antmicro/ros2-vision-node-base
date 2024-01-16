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

        # Allocate memory for inputs and outputs
        # Define input and output specifications
        self.input_specs = []
        self.output_specs = []
        self.allocations = []
        for i in range(engine.num_bindings):
            is_input = False
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = trt_utils.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.batch_size = shape[0]
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
        predictions = []
        for output_spec in self.output_specs:
            shape, dtype = output_spec["shape"], output_spec["dtype"]
            predictions.append(np.zeros(shape, dtype))

        # Move input data to device
        for x, input_spec in zip(X, self.input_specs):
            trt_utils.memcpy_host_to_device(
                input_spec["allocation"], np.ascontiguousarray(x)
            )

        # Run inference
        self.trt_context.execute_v2(self.allocations)

        # Move output data to host
        for i, output in enumerate(predictions):
            trt_utils.memcpy_device_to_host(
                output, self.output_specs[i]["allocation"]
            )

        return predictions

    def cleanup(self):
        del self.trt_context
        for allocation in self.allocations:
            trt_utils.cuda_call(cudart.cudaFree(allocation))
        trt_utils.cuda_call(cudart.cudaDeviceReset())
