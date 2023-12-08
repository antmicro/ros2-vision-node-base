#!/usr/bin/env python3

# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""YOLACT ROS2 node implementation."""

import rclpy
from cvnode_base.cvnode_base import BaseCVNode
from cvnode_base.utils import imageToMat
from gc import collect
from kenning_computer_vision_msgs.msg import SegmentationMsg, MaskMsg, BoxMsg
from kenning.modelwrappers.instance_segmentation.yolact import YOLACT
from pathlib import Path
import traceback


class YOLACTOnnx:
    def __init__(self, node: BaseCVNode):
        """
        Initialize YOLACTOnnx wrapper.

        Parameters
        ----------
        node : BaseCVNode
            ROS2 node.
        """
        self.node = node    # ROS2 node

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool :
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.onnx import ONNXRuntime
        except ImportError:
            self.node.get_logger().error('Cannot import ONNXRuntime')
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter('model_path').value)
        if not model_path.exists():
            self.node.get_logger().error(f'File {model_path} does not exist')
            return False
        self.model = YOLACT(model_path, None)
        self.runtime = ONNXRuntime(
            model_path,
            execution_providers=["CPUExecutionProvider",
                                 "GPUExecutionProvider"],
            disable_performance_measurements=True
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTTVM:
    def __init__(self, node: BaseCVNode):
        """
        Initialize YOLACTTVM wrapper.

        Parameters
        ----------
        node : BaseCVNode
            ROS2 node.
        """
        self.node = node        # ROS2 node
        self.node.declare_parameter('device', rclpy.Parameter.Type.STRING)

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool :
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.tvm import TVMRuntime
        except ImportError:
            self.node.get_logger().error('Cannot import TVMRuntime')
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter('model_path').value)
        if not model_path.exists():
            self.node.get_logger().error(f'File {model_path} does not exist')
            return False
        device = self.node.get_parameter('device').value
        if not device:
            self.node.get_logger().error(
                    'Please specify device for TVM runtime'
            )
            return False
        self.model = YOLACT(model_path, None)
        self.runtime = TVMRuntime(
            model_path,
            contextname=device,
            disable_performance_measurements=True
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTTFLite:
    def __init__(self, node: BaseCVNode):
        """
        Initialize YOLACTTFLite wrapper.

        Parameters
        ----------
        node : BaseCVNode
            ROS2 node.
        """
        self.node = node        # ROS2 node

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool :
            True if preparation was successful, False otherwise.
        """
        try:
            from kenning.runtimes.tflite import TFLiteRuntime
        except ImportError:
            self.node.get_logger().error('Cannot import TFLiteRuntime')
            self.node.get_logger().error(str(traceback.format_exc()))
            return False

        model_path = Path(self.node.get_parameter('model_path').value)
        if not model_path.exists():
            self.node.get_logger().error(f'File {model_path} does not exist')
            return False
        self.model = YOLACT(model_path, None)
        self.runtime = TFLiteRuntime(
            model_path,
            disable_performance_measurements=True
        )
        ret = self.runtime.prepare_local()
        return ret


class YOLACTNode(BaseCVNode):
    backends = {
        'tvm': YOLACTTVM,
        'tflite': YOLACTTFLite,
        'onnxruntime': YOLACTOnnx,
    }

    def __init__(self, node_name: str):
        """
        Initialize YOLACTNode.

        Parameters
        ----------
        node_name : str
            Name of the node.
        """
        self.yolact = None  # Wrapper for YOLACT model with runtime
        super().__init__(node_name=node_name)
        self.declare_parameter('backend', rclpy.Parameter.Type.STRING)
        self.declare_parameter('model_path', rclpy.Parameter.Type.STRING)

    def prepare(self):
        backend = self.get_parameter('backend').value
        if backend not in self.backends:
            self.get_logger().error(f'Backend {backend} is not supported')
            return False
        self.yolact = self.backends[backend](self)
        return self.yolact.prepare()

    def run_inference(self, X):
        result = []
        for frame in X:
            x = imageToMat(frame, 'rgb8').transpose(2, 0, 1)
            x = self.yolact.model.preprocess_input([x])
            self.yolact.runtime.load_input([x])
            self.yolact.runtime.run()
            preds = self.yolact.runtime.extract_output()
            preds = self.yolact.model.postprocess_outputs(preds)

            msg = SegmentationMsg()
            msg._frame = frame
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
            result.append(msg)
        return result

    def cleanup(self):
        del self.yolact
        collect()


if __name__ == '__main__':
    rclpy.init()
    node = YOLACTNode('yolact_node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()