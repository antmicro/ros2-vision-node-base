#!/usr/bin/env python3

# Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Dict

import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import detection_utils
from detectron2.export import TracingAdapter
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.file_io import PathManager


def export_tracing(model: GeneralizedRCNN,
                   inputs: Dict[str, torch.Tensor],
                   output_path: str):
    """
    Export model to TorchScript format.

    Parameters
    ----------
    model : GeneralizedRCNN
        Model to export.
    inputs : Dict[str, torch.Tensor]
        Sample inputs for model containing only 'image' key.
    output_path : str
        Path to directory where model will be saved.
    """
    def inference(model, inputs):
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]

    traceable_model = TracingAdapter(model, inputs, inference)
    ts_model = torch.jit.trace(traceable_model, (inputs[0]['image'],))
    with PathManager.open(os.path.join(output_path, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    return


def get_sample_inputs(image_path: str,
                      cfg: CfgNode) -> Dict[str, torch.Tensor]:
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
    List[Dict[str, torch.Tensor]] :
        List containing one dictionary with 'image' key and image tensor.
    """
    original_image = detection_utils.read_image(
            image_path,
            format=cfg.INPUT.FORMAT
    )
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST
    )
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    return [{'image': image}]


def prepare_config() -> CfgNode:
    """
    Prepare model configuration.

    Returns
    -------
    CfgNode :
        Model configuration.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
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
            help="Path to image file used for model tracing"
    )
    parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Path to directory where model will be saved"
    )

    args = parser.parse_args()
    image_path = args.image
    output_path = args.output

    assert os.path.isfile(image_path), "Image file does not exist"
    assert os.path.isdir(output_path), "Output directory does not exist"

    cfg = prepare_config()
    sample_inputs = get_sample_inputs(image_path, cfg)

    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    export_tracing(model, sample_inputs, output_path)

    print(f"Model exported successfully to {os.path.join(output_path, 'model.ts')}")    # noqa: E501
