# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""CVNode with Mask R-CNN model from Detectron2 framework."""

import csv
import os
from gc import collect
from typing import Dict, List

import cv2
import numpy as np
import rclpy
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.modeling import build_model
from kenning_computer_vision_msgs.msg import BoxMsg, MaskMsg, SegmentationMsg
from sensor_msgs.msg import Image
from torch import as_tensor

from cvnode_base.cvnode_base import BaseCVNode


class MaskRCNNDetectronNode(BaseCVNode):
    """The Detectron2 implementation of a Mask R-CNN model in a CVNode."""

    def __init__(self):
        """Initialize node."""
        super().__init__(node_name='mask_rcnn_detectron_node')
        self.declare_parameter('class_names_path', rclpy.Parameter.Type.STRING)

    def run_inference(self, X: List[Image]) -> List[SegmentationMsg]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : List[Image]
            List of input image messages.

        Returns
        -------
        List[SegmentationMsg] :
            List of postprocessed segmentation messages.
        """
        input_data = self.preprocess(X)
        predictions = self.predict(input_data)
        input_data = None
        return self.postprocess(predictions, X)

    def prepare(self) -> bool:
        """
        Prepare node for execution.

        Returns
        -------
        bool :
            True if the node is ready for execution, False otherwise.
        """
        class_names_path = self.get_parameter('class_names_path').value
        if not os.path.exists(class_names_path):
            self.logger.error(f'File {class_names_path} does not exist')
            return False
        with open(class_names_path, 'r') as f:
            reader = csv.reader(f)
            reader.__next__()
            self.classes = tuple([row[0] for row in reader])
            if not self.classes:
                self.logger.error(f'File {class_names_path} is empty')
                return False

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.model = build_model(cfg.clone())
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST)
        return True

    def preprocess(self, X: List[Image]) -> List[Dict]:
        """
        Preprocess input data.

        Parameters
        ----------
        X : List[Image]
            List of input image messages.

        Returns
        -------
        List[Dict] :
            Preprocessed data compatible with the model.
        """
        Y = []
        for msg in X:
            img = self.convert_image_format(msg.data, msg.encoding)
            height = msg.height
            width = msg.width
            img = np.reshape(img, (height, width, -1))
            augmented = self.aug.get_transform(img).apply_image(img)
            augmented = as_tensor(
                augmented.astype('float32').transpose(2, 0, 1))
            Y.append({'image': augmented, 'height': height, 'width': width})
        return Y

    def predict(self, X: List[Dict]) -> List[Dict]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : List[Dict]
            Input data.

        Returns
        -------
        List[Dict] :
            Model predictions.
        """
        return [self.model([x])[0] for x in X]

    def postprocess(self, Y: List[Dict], images: List[Image]
                    ) -> List[SegmentationMsg]:
        """
        Postprocess model predictions.

        Parameters
        ----------
        Y : List[Dict]
            Model predictions.
        images : List[Image]
            List of input image messages.

        Returns
        -------
        List[SegmentationMsg] :
            Postprocessed model predictions in the form of SegmentationMsg.
        """
        X = []
        for i, prediction in enumerate(Y):
            msg = SegmentationMsg()
            msg._frame = images[i]
            prediction = prediction['instances']
            scores = prediction.scores.cpu().detach().numpy()

            for mask_np in prediction.pred_masks.cpu().detach().numpy():
                mask = MaskMsg()
                mask._dimension = [mask_np.shape[0], mask_np.shape[1]]
                mask._data = mask_np.flatten().astype('uint8')
                msg._masks.append(mask)

            boxes = prediction.pred_boxes.tensor
            boxes[:, 0] /= prediction.image_size[1]
            boxes[:, 1] /= prediction.image_size[0]
            boxes[:, 2] /= prediction.image_size[1]
            boxes[:, 3] /= prediction.image_size[0]

            for box_np in boxes.cpu().detach().numpy():
                box = BoxMsg()
                box._xmin = float(box_np[0])
                box._ymin = float(box_np[1])
                box._xmax = float(box_np[2])
                box._ymax = float(box_np[3])
                msg._boxes.append(box)

            labels = prediction.pred_classes.cpu().detach().numpy()

            msg._scores = scores
            msg._classes = [self.classes[x] for x in labels]
            X.append(msg)
        return X

    def cleanup(self):
        """
        Cleanup allocated resources used by the node.
        """
        del self.model
        del self.aug
        collect()

    def convert_image_format(self, image: np.ndarray, encoding: str
                             ) -> np.ndarray:
        """
        Convert image to BGR format.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        encoding : str
            Image encoding.

        Returns
        -------
        np.ndarray :
            Image in BGR format.
        """
        if (encoding in ['bgr8', '8UC3']):
            return np.asarray(image, dtype=np.uint8)
        elif (encoding == 'rgb8'):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif (encoding in ['mono8', '8UC1']):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif (encoding in ['bgra8', '8UC4']):
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif (encoding == 'rgba8'):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
