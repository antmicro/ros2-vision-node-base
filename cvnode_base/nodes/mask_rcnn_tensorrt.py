#!/usr/bin/env python3

# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""CVNode with Mask R-CNN model in a TensorRT serialized engine format."""

import csv
import os
from gc import collect
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from kenning_computer_vision_msgs.msg import BoxMsg, MaskMsg, SegmentationMsg
from sensor_msgs.msg import Image

from cvnode_base.core.tensorrt_base import CVNodeTensorRTBase
from cvnode_base.utils.image import imageToMat


class MaskRCNNTensorRTNode(CVNodeTensorRTBase):
    """The TensorRT implementation of a Mask R-CNN model in a CVNode."""

    def __init__(self):
        super().__init__(node_name="mask_rcnn_tensorrt_node")
        self.declare_parameter("class_names_path", rclpy.Parameter.Type.STRING)

    def run_inference(self, X):
        input_data = self.preprocess(X.frame)
        prediction = self.predict(input_data)
        result = self.postprocess(prediction, X.frame)
        return True, result

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
        ret = super().prepare()
        self.input_shape = self.input_specs[0]["shape"]
        self.input_shape = (self.input_shape[-2], self.input_shape[-1])
        return ret

    def preprocess(self, frame: Image) -> List[np.ndarray]:
        """
        Preprocess input data.

        Parameters
        ----------
        frame : Image
            Input image message.

        Returns
        -------
        List[np.ndarray]
            Resized image compatible with the model input shape.
        """
        img = imageToMat(frame, "rgb8")
        img = cv2.resize(
            img,
            self.input_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)
        img = img.astype(self.input_specs[0]["dtype"])
        return [img]

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
        keep = Y[0].squeeze()
        boxes, classes, masks, scores = (
            Y[1].squeeze()[:keep],
            Y[3].squeeze()[:keep],
            Y[4].squeeze()[:keep],
            Y[2].squeeze()[:keep],
        )
        if masks.ndim == 2:
            masks = np.expand_dims(masks, 0)

        keep = np.where(scores > 0.3)
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        scores = scores[keep]

        keep = []
        for idx, box_np in enumerate(boxes):
            box = BoxMsg()
            if box_np.min() < 0 or box_np.max() > 1:
                continue
            keep.append(idx)
            box._xmin = float(box_np[0])
            box._ymin = float(box_np[1])
            box._xmax = float(box_np[2])
            box._ymax = float(box_np[3])
            msg._boxes.append(box)
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Paste masks to the original image
        masks = self.resize_masks(masks, boxes, (frame.height, frame.width))
        for mask_np in masks:
            mask = MaskMsg()
            mask._dimension = [mask_np.shape[0], mask_np.shape[1]]
            mask._data = (mask_np * 255.0).flatten().astype("uint8")
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
        super().cleanup()
        del self.input_shape
        del self.classes
        collect()


def main(args=None):
    """Run the MaskRCNNTensorRTNode node."""
    rclpy.init(args=args)
    node = MaskRCNNTensorRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
