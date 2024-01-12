#!/usr/bin/env python3

# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""CVNode with Mask R-CNN model in ONNX format."""

import csv
import os
import traceback
from gc import collect
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from kenning_computer_vision_msgs.msg import BoxMsg, MaskMsg, SegmentationMsg
from sensor_msgs.msg import Image
from torch.cuda import empty_cache

from cvnode_base.cvnode_base import BaseCVNode
from cvnode_base.utils.image import imageToMat


class MaskRCNNONNXNode(BaseCVNode):
    """The ONNX implementation of a Mask R-CNN model in a CVNode."""

    def __init__(self):
        super().__init__(node_name="mask_rcnn_onnx_node")
        self.declare_parameter("class_names_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("model_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("device", rclpy.Parameter.Type.STRING)

    def run_inference(self, X: List[Image]) -> List[SegmentationMsg]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : List[Image]
            List of input image messages.

        Returns
        -------
        List[SegmentationMsg]
            List of postprocessed segmentation messages.
        """
        result = []
        for frame in X:
            input_data = self.preprocess(frame)
            prediction = self.predict(input_data)
            result.append(self.postprocess(prediction, frame))
            empty_cache()
        return result

    def prepare(self) -> bool:
        """
        Prepare node for execution.

        Returns
        -------
        bool
            True if the node is ready for execution, False otherwise.
        """
        # Load class names
        class_names_path = self.get_parameter("class_names_path").value
        if not os.path.exists(class_names_path):
            self.get_logger().error(f"File {class_names_path} does not exist")
            return False
        with open(class_names_path, "r") as f:
            reader = csv.reader(f)
            reader.__next__()
            self.classes = tuple([row[0] for row in reader])
            if not self.classes:
                self.get_logger().error(f"File {class_names_path} is empty")
                return False

        # Load model
        providers = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }
        device = self.get_parameter("device").value
        if device not in providers:
            self.get_logger().error(f"Device '{device}' is not supported")
            self.get_logger().error(
                f"Supported devices: {list(providers.keys())}"
            )
            return False
        model_path = self.get_parameter("model_path").value
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model '{model_path}' does not exist")
            return False

        try:
            self.model = ort.InferenceSession(
                model_path, providers=[providers[device]]
            )
        except (TypeError, ValueError, RuntimeError):
            self.get_logger().error("Could not load ONNX model")
            self.get_logger().error(str(traceback.format_exc()))
            return False

        # Set input/output metadata
        input_metadata = self.model.get_inputs()[0]
        self.input_name = input_metadata.name
        self.input_shape = input_metadata.shape
        self.output_names = [x.name for x in self.model.get_outputs()]

        # Warmup the model
        if device == "cuda":
            for _ in range(5):
                noise = (
                    np.random.randint(
                        0, high=256, size=np.prod(self.input_shape)
                    )
                    .reshape(self.input_shape)
                    .astype(np.uint8)
                )
                self.model.run(self.output_names, {self.input_name: noise})
        return True

    def preprocess(self, frame: Image) -> np.array:
        """
        Preprocess input data.

        Parameters
        ----------
        frame : Image
            Input image message.

        Returns
        -------
        np.array
            Resized image compatible with the model input shape.
        """
        img = imageToMat(frame, "rgb8")
        image_shape = (self.input_shape[-1], self.input_shape[-2])
        img = cv2.resize(
            img,
            image_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        img = img.transpose((2, 0, 1))
        return img

    def predict(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        List[np.ndarray]
            List of predictions.
        """
        return self.model.run(self.output_names, {self.input_name: X})

    def postprocess(
        self, Y: List[np.ndarray], frame: Image
    ) -> SegmentationMsg:
        """
        Postprocess model predictions.

        Parameters
        ----------
        Y : List[np.ndarray]
            Model predictions.
        frame : Image
            Input image message.

        Returns
        -------
        SegmentationMsg
            Postprocessed model predictions in the form of SegmentationMsg.
        """
        msg = SegmentationMsg()
        msg._frame = frame
        boxes, classes, masks, scores = Y[0], Y[1], Y[2].squeeze(), Y[3]
        if masks.ndim == 2:
            masks = np.expand_dims(masks, 0)

        # Scale boxes to 0-1 range
        boxes[:, 0], boxes[:, 2] = (
            boxes[:, 0] / self.input_shape[-1],
            boxes[:, 2] / self.input_shape[-1],
        )
        boxes[:, 1], boxes[:, 3] = (
            boxes[:, 1] / self.input_shape[-2],
            boxes[:, 3] / self.input_shape[-2],
        )
        for box_np in boxes:
            box = BoxMsg()
            box._xmin = float(box_np[0])
            box._ymin = float(box_np[1])
            box._xmax = float(box_np[2])
            box._ymax = float(box_np[3])
            msg._boxes.append(box)

        # Paste masks to the original image
        masks = self.resize_masks(masks, boxes, (frame.height, frame.width))
        for mask_np in masks:
            mask = MaskMsg()
            mask._dimension = [mask_np.shape[0], mask_np.shape[1]]
            mask._data = mask_np.flatten().astype("uint8")
            msg._masks.append(mask)

        msg._classes = [self.classes[int(x)] for x in classes]
        msg._scores = [float(x) for x in scores]
        return msg

    def resize_masks(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        image_shape: Tuple[int, int],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Resize masks to the original image shape.

        Parameters
        ----------
        masks : np.ndarray
            Masks to be resized.
        boxes : np.ndarray
            Bounding boxes of the masks.
        image_shape : Tuple[int, int]
            Original image shape.
        threshold : float
            Threshold for the masks. Default: 0.5.

        Returns
        -------
        np.ndarray
            Resized masks.
        """
        resized_masks = np.zeros((len(masks), *image_shape), dtype=np.uint8)
        boxes[:, 0], boxes[:, 2] = (
            boxes[:, 0] * image_shape[1],
            boxes[:, 2] * image_shape[1],
        )
        boxes[:, 1], boxes[:, 3] = (
            boxes[:, 1] * image_shape[0],
            boxes[:, 3] * image_shape[0],
        )
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            int_box = box.astype(np.int32)
            mask_shape = (
                max(int_box[2] - int_box[0], 1),
                max(int_box[3] - int_box[1], 1),
            )
            mask = cv2.resize(mask, mask_shape)
            mask = (mask > threshold).astype(np.uint8)
            resized_masks[
                i, int_box[1] : int_box[3], int_box[0] : int_box[2]
            ] = mask
        return resized_masks

    def cleanup(self):
        """Cleanup allocated resources used by the node."""
        del self.model
        del self.input_name
        del self.input_shape
        del self.output_names
        del self.classes
        collect()


def main(args=None):
    """Run the MaskRCNNONNXNode node."""
    rclpy.init(args=args)
    node = MaskRCNNONNXNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
