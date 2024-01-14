#!/usr/bin/env python3

# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""CVNode with YOLACT model in a TensorRT serialized engine format."""

import csv
import os
from gc import collect
from typing import List

import numpy as np
import rclpy
from kenning.modelwrappers.instance_segmentation.yolact import YOLACT
from kenning_computer_vision_msgs.msg import BoxMsg, MaskMsg, SegmentationMsg
from sensor_msgs.msg import Image

from cvnode_base.core.tensorrt_base import CVNodeTensorRTBase
from cvnode_base.utils.image import imageToMat


class YOLACTTensorRTNode(CVNodeTensorRTBase):
    """The TensorRT implementation of a YOLACT model in a CVNode."""

    def __init__(self):
        super().__init__(node_name="mask_rcnn_tensorrt_node")
        self.declare_parameter("class_names_path", rclpy.Parameter.Type.STRING)

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

        class DummyDataset:
            def __init__(self, classes):
                self.classes = classes

            def get_class_names(self):
                return self.classes

        # We employ Kenning's YOLACT wrapper for fair comparison
        # of the models performance
        self.yolact_wrapper = YOLACT(None, DummyDataset(self.classes))

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
        img = self.yolact_wrapper.preprocess_input(
            [np.transpose(img, (2, 0, 1))]
        )
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
        msg._frame = frame
        proto = Y[0]
        loc = Y[1]
        mask = Y[2]
        conf = Y[3]
        priors = Y[4]
        Y = [loc, conf, mask, priors, proto]
        postprocessed = self.yolact_wrapper.postprocess_outputs(Y)
        if postprocessed:
            for segmentation in postprocessed[0]:
                box = BoxMsg()
                box._xmin = float(segmentation.xmin)
                box._ymin = float(segmentation.ymin)
                box._xmax = float(segmentation.xmax)
                box._ymax = float(segmentation.ymax)
                msg._boxes.append(box)

                mask = MaskMsg()
                mask_arr = segmentation.mask
                mask._dimension = mask_arr.shape
                mask._data = mask_arr.flatten()
                msg._masks.append(mask)

                msg._scores.append(float(segmentation.score))
                msg._classes.append(segmentation.clsname)
        return msg

    def cleanup(self):
        """Cleanup allocated resources used by the node."""
        super().cleanup()
        del self.input_shape
        del self.classes
        collect()


def main(args=None):
    """Run the YOLACTTensorRTNode node."""
    rclpy.init(args=args)
    node = YOLACTTensorRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
