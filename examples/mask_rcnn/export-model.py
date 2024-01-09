#!/usr/bin/env python3

# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Export Mask R-CNN model to TorchScript or ONNX format."""

# NOTE: Was verified for detectron2@8c4a333ceb8df05348759443d0206302485890e0

import argparse
import os
from typing import Dict, List

import cv2
import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import detection_utils
from detectron2.export import TracingAdapter
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.file_io import PathManager


def export_tracing(
    model: GeneralizedRCNN,
    inputs: Dict[str, torch.Tensor],
    output_path: str,
    format: str = "torchscript",
):
    """
    Export model to desired format.

    Parameters
    ----------
    model : GeneralizedRCNN
        Model to export.
    inputs : Dict[str, torch.Tensor]
        Sample inputs for model containing only 'image' key.
    output_path : str
        Path to directory where model will be saved.
    format : str
        Format in which model will be exported.
    """

    def inference(model, inputs):
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]

    traceable_model = TracingAdapter(model, inputs, inference)
    if format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (inputs[0]["image"],))
        model_path = os.path.join(output_path, "model.ts")
        with PathManager.open(model_path, "wb") as f:
            torch.jit.save(ts_model, f)
    elif format == "onnx":
        model_path = os.path.join(output_path, "model.onnx")
        with PathManager.open(model_path, "wb") as f:
            output_names = ["output_0", "output_1", "output_2", "output_3"]
            torch.onnx.export(
                traceable_model,
                (inputs[0]["image"],),
                f,
                opset_version=11,
                export_params=True,
                output_names=output_names,
                input_names=["input"],
            )


def get_sample_inputs(
    image_path: str, cfg: CfgNode
) -> List[Dict[str, torch.Tensor]]:
    """
    Prepare sample inputs for model.

    Parameters
    ----------
    image_path : str
        Path to image file.
    cfg : CfgNode
        Model configuration.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List containing one dictionary with 'image' key and image tensor.
    """
    original_image = detection_utils.read_image(
        image_path, format=cfg.INPUT.FORMAT
    )
    resized_image = cv2.resize(
        original_image, (1202, 800), interpolation=cv2.INTER_LINEAR
    )
    aug = T.ResizeShortestEdge([800, 800], 1202)
    image = aug.get_transform(resized_image).apply_image(resized_image)
    image = torch.as_tensor(image.astype("uint8").transpose(2, 0, 1))
    return [{"image": image}]


def prepare_config(weights: str, num_classes: int) -> CfgNode:
    """
    Prepare model configuration.

    Parameters
    ----------
    weights : str
        Path to file containing model weights.
    num_classes : int
        Number of classes used in the model.

    Returns
    -------
    CfgNode
        Model configuration.

    Raises
    ------
    FileNotFoundError
        If weights file does not exist.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    if weights == "COCO":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    else:
        if not os.path.isfile(weights):
            raise FileNotFoundError("Weights file does not exist")
        cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Mask R-CNN model to TorchScript"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file used for model tracing",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to directory where model will be saved",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["torchscript", "onnx"],
        help="Export method",
    )

    # Optional arguments
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        default="COCO",
        help="Path to file containing model weights. "
        "If set to 'COCO', the weights will be loaded from the model zoo",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=False,
        default=80,
        help="Number of classes used in the model",
    )

    args = parser.parse_args()
    image_path = args.image
    output_path = args.output

    assert os.path.isfile(image_path), "Image file does not exist"
    assert os.path.isdir(output_path), "Output directory does not exist"

    cfg = prepare_config(args.weights, args.num_classes)
    sample_inputs = get_sample_inputs(image_path, cfg)

    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    export_tracing(model, sample_inputs, output_path, args.method)

    print("Model exported successfully")
