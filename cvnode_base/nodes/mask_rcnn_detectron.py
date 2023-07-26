"""ROS2 node for MaskRCNN inference using Detectron2 framework."""

from cvnode_base.cvnode_base import BaseCVNode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from typing import List, Dict

from cvnode_msgs.msg import SegmentationMsg, MaskMsg, BoxMsg
from sensor_msgs.msg import Image

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch


class MaskRCNNDetectronNode(BaseCVNode):
    """The MaskRCNN ROS2 node implemented using Detectron2 framework."""
    classes = ('person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife',
               'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush',
               )

    def __init__(self):
        """
        Initialize the MaskRCNN node.
        """
        super().__init__(node_name="mask_rcnn_detectron_node")

    def prepare(self) -> bool:
        """
        Prepare node for execution.

        Returns
        -------
        bool :
            True if the node is ready for execution, False otherwise.
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.model = build_model(self.cfg.clone())
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST)
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
        self.frames = []
        for msg in X:
            img = self.convert_image_format(msg.data, msg.encoding)
            self.frames.append(msg)
            height = msg.height
            width = msg.width
            img = np.reshape(img, (height, width, -1))
            augmented = self.aug.get_transform(img).apply_image(img)
            augmented = torch.as_tensor(
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

    def postprocess(self, Y: List[Dict]) -> List[SegmentationMsg]:
        """
        Postprocess model predictions.

        Parameters
        ----------
        Y : List[Dict]
            Model predictions.

        Returns
        -------
        List[SegmentationMsg] :
            Postprocessed model predictions in the form of SegmentationMsg.
        """
        X = []
        for i, prediction in enumerate(Y):
            msg = SegmentationMsg()
            msg._frame = self.frames[i]
            prediction = prediction['instances']
            scores = prediction.scores.cpu().detach().numpy()

            for mask_np in prediction.pred_masks.cpu().detach().numpy():
                mask = MaskMsg()
                mask._dimension = [mask_np.shape[0], mask_np.shape[1]]
                mask._data = mask_np.flatten().astype('uint8')
                msg._masks.append(mask)

            for box_np in prediction.pred_boxes.tensor.cpu(
                    ).detach().numpy():
                box = BoxMsg()
                box._xmin = float(box_np[0] / prediction.image_size[1])
                box._ymin = float(box_np[1] / prediction.image_size[0])
                box._xmax = float(box_np[2] / prediction.image_size[1])
                box._ymax = float(box_np[3] / prediction.image_size[0])
                msg._boxes.append(box)

            labels = prediction.pred_classes.cpu().detach().numpy()

            msg._scores = scores
            msg._classes = [self.classes[x] for x in labels]
            X.append(msg)
        return X

    def cleanup(self):
        """
        Cleanup allocated resources.
        """
        if (self.model):
            del self.model
            self.model = None
        if (self.cfg):
            del self.cfg
            self.cfg = None
        if (self.aug):
            del self.aug
            self.aug = None

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
        if (encoding in ["bgr8", "8UC3"]):
            return np.asarray(image, dtype=np.uint8)
        elif (encoding == "rgb8"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif (encoding in ["mono8", "8UC1"]):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif (encoding in ["bgra8", "8UC4"]):
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif (encoding == "rgba8"):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
