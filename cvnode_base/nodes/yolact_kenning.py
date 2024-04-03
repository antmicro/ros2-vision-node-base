#!/usr/bin/env python3

# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""YOLACT ROS2 node implementation."""

import traceback
from gc import collect
from pathlib import Path

import rclpy
from kenning.modelwrappers.instance_segmentation.yolact import YOLACT
from kenning_computer_vision_msgs.msg import BoxMsg, MaskMsg, SegmentationMsg

from cvnode_base.core.cvnode_base import BaseCVNode
from cvnode_base.utils.image import imageToMat


class YOLACTOnnx:
    """
    ONNX runtime wrapper for YOLACT model.
    """

    def __init__(self, node: BaseCVNode):
        self.node = node  # ROS2 node
        self.node.declare_parameter("device", rclpy.Parameter.Type.STRING)

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.onnx import ONNXRuntime
        except ImportError:
            self.node.get_logger().error("Cannot import ONNXRuntime")
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter("model_path").value)
        if not model_path.exists():
            self.node.get_logger().error(f"File {model_path} does not exist")
            return False
        self.model = YOLACT(model_path, None, top_k=100, score_threshold=0.3)

        device = self.node.get_parameter("device").value
        if not device:
            self.node.get_logger().error(
                "Please specify device for TVM runtime"
            )
            return False
        if device == "cpu":
            execution_providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            self.node.get_logger().error(
                f"Device {device} is not supported by ONNX runtime"
            )
            return False
        self.runtime = ONNXRuntime(
            model_path,
            execution_providers=execution_providers,
            disable_performance_measurements=True,
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTTVM:
    """
    TVM runtime wrapper for YOLACT model.
    """

    def __init__(self, node: BaseCVNode):
        self.node = node  # ROS2 node
        self.node.declare_parameter("device", rclpy.Parameter.Type.STRING)

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.tvm import TVMRuntime
        except ImportError:
            self.node.get_logger().error("Cannot import TVMRuntime")
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter("model_path").value)
        if not model_path.exists():
            self.node.get_logger().error(f"File {model_path} does not exist")
            return False
        device = self.node.get_parameter("device").value
        if not device:
            self.node.get_logger().error(
                "Please specify device for TVM runtime"
            )
            return False
        self.model = YOLACT(model_path, None, top_k=100, score_threshold=0.3)
        self.runtime = TVMRuntime(
            model_path,
            contextname=device,
            disable_performance_measurements=True,
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTTFLite:
    """
    TFLite runtime wrapper for YOLACT model.
    """

    def __init__(self, node: BaseCVNode):
        self.node = node  # ROS2 node

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.tflite import TFLiteRuntime
        except ImportError:
            self.node.get_logger().error("Cannot import TFLiteRuntime")
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter("model_path").value)
        if not model_path.exists():
            self.node.get_logger().error(f"File {model_path} does not exist")
            return False
        self.model = YOLACT(model_path, None, top_k=100, score_threshold=0.3)
        self.runtime = TFLiteRuntime(
            model_path, disable_performance_measurements=True
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTNode(BaseCVNode):
    """
    ROS2 node for YOLACT model.
    """

    backends = {
        "tvm": YOLACTTVM,
        "tflite": YOLACTTFLite,
        "onnxruntime": YOLACTOnnx,
    }

    def __init__(self, node_name: str):
        self.yolact = None  # Wrapper for YOLACT model with runtime
        super().__init__(node_name=node_name)
        self.declare_parameter("backend", rclpy.Parameter.Type.STRING)
        self.declare_parameter("model_path", rclpy.Parameter.Type.STRING)

    def prepare(self):
        backend = self.get_parameter("backend").value
        if backend not in self.backends:
            self.get_logger().error(f"Backend {backend} is not supported")
            return False
        self.yolact = self.backends[backend](self)
        return self.yolact.prepare()

    def run_inference(self, X):
        x = imageToMat(X.frame, "rgb8").transpose(2, 0, 1)
        x = self.yolact.model.preprocess_input([x])
        self.yolact.runtime.load_input([x])
        self.yolact.runtime.run()
        preds = self.yolact.runtime.extract_output()
        preds = self.yolact.model.postprocess_outputs(preds)

        msg = SegmentationMsg()
        if preds:
            for y in preds[0]:
                box = BoxMsg()
                box._xmin = float(y.xmin)
                box._xmax = float(y.xmax)
                box._ymin = float(y.ymin)
                box._ymax = float(y.ymax)
                msg._boxes.append(box)
                msg._scores.append(y.score)
                mask = MaskMsg()
                mask._data = y.mask.flatten()
                mask._dimension = [y.mask.shape[0], y.mask.shape[1]]
                msg._masks.append(mask)
                msg._classes.append(y.clsname)
        return True, msg

    def cleanup(self):
        del self.yolact
        collect()


if __name__ == "__main__":
    rclpy.init()
    node = YOLACTNode("yolact_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
