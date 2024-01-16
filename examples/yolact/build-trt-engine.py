#!/usr/bin/env python3

# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Export ONNX model to TensorRT engine."""

import argparse
import os

import tensorrt as trt


class EngineBuilder:
    """
    TensorRT engine builder from ONNX model files.
    """

    def __init__(self, max_memory: int = 8):
        self.trt_logger = trt.Logger(trt.Logger.INFO)

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = max_memory * (2**30)

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path: str):
        """
        Creates the TensorRT network definition from ONNX file.

        Parameters
        ----------
        onnx_path : str
            The path to the ONNX file to load.

        Raises
        ------
        ValueError
            If failed to parse the ONNX file.
        """
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )  # noqa: E501
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print(f"Failed to load ONNX file: {onnx_path}")
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise ValueError("Failed to parse the ONNX file.")

        self.batch_size = self.network.get_input(0).shape[0]
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path: str):
        """
        Builds the TensorRT engine and serialize it to a file.

        Parameters
        ----------
        engine_path : str
            The path to the output engine file.

        Raises
        ------
        ValueError
            If failed to build the engine.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)

        engine_bytes = None
        try:
            engine_bytes = self.builder.build_serialized_network(
                self.network, self.config
            )
        except AttributeError:
            engine = self.builder.build_engine(self.network, self.config)
            engine_bytes = engine.serialize()
            del engine

        if not engine_bytes:
            raise ValueError("Failed to build the engine.")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--onnx",
        type=str,
        required=True,
        help="The input ONNX model file to load",
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        required=False,
        help="The output path for the TRT engine",
    )
    parser.add_argument(
        "-m",
        "--max-memory",
        type=int,
        required=False,
        default=1,
        help="The max memory workspace size to allow in Gb",
    )
    args = parser.parse_args()
    builder = EngineBuilder(args.max_memory)
    builder.create_network(args.onnx)
    builder.create_engine(args.engine)
