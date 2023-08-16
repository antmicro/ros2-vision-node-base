// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cvnode_base/cvnode_base.hpp>
#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace cvnode_base
{

/**
 * Container for Mask R-CNN outputs.
 */
struct MaskRCNNOutputs
{
    at::Tensor boxes;   ///< Bounding boxes
    at::Tensor classes; ///< Class IDs
    at::Tensor masks;   ///< Masks
    at::Tensor scores;  ///< Scores
};

/**
 * TorchScript implementation of Mask R-CNN model in a CVNode.
 *
 * @param model_path ROS 2 string parameter with path to model TorchScript file.
 */
class MaskRCNNTorchScript : public BaseCVNode
{
private:
    /**
     * Paste masks to the ROI of the image.
     *
     * @param mask Mask to paste to the image.
     * @param box Bounding box of the ROI.
     * @param height Height of the image.
     * @param width Width of the image.
     *
     * @return Pasted mask.
     */
    cv::Mat paste_mask(const at::Tensor &mask, const at::Tensor &box, const int height, const int width);

    std::string model_path;                                 ///< Path to TorchScript file
    torch::jit::script::Module model;                       ///< TorchScript model
    c10::Device device = c10::Device(c10::DeviceType::CPU); ///< Device to run inference on

    std::vector<sensor_msgs::msg::Image> frames; ///< Input images
    std::vector<c10::IValue> inputs;             ///< Preprocessed input images
    std::vector<MaskRCNNOutputs> predictions;    ///< Inference outputs

    /// Class names
    std::vector<std::string> class_names{"person",        "bicycle",      "car",
                                         "motorcycle",    "airplane",     "bus",
                                         "train",         "truck",        "boat",
                                         "traffic light", "fire hydrant", "stop sign",
                                         "parking meter", "bench",        "bird",
                                         "cat",           "dog",          "horse",
                                         "sheep",         "cow",          "elephant",
                                         "bear",          "zebra",        "giraffe",
                                         "backpack",      "umbrella",     "handbag",
                                         "tie",           "suitcase",     "frisbee",
                                         "skis",          "snowboard",    "sports ball",
                                         "kite",          "baseball bat", "baseball glove",
                                         "skateboard",    "surfboard",    "tennis racket",
                                         "bottle",        "wine glass",   "cup",
                                         "fork",          "knife",        "spoon",
                                         "bowl",          "banana",       "apple",
                                         "sandwich",      "orange",       "broccoli",
                                         "carrot",        "hot dog",      "pizza",
                                         "donut",         "cake",         "chair",
                                         "couch",         "potted plant", "bed",
                                         "dining table",  "toilet",       "tv",
                                         "laptop",        "mouse",        "remote",
                                         "keyboard",      "cell phone",   "microwave",
                                         "oven",          "toaster",      "sink",
                                         "refrigerator",  "book",         "clock",
                                         "vase",          "scissors",     "teddy bear",
                                         "hair drier",    "toothbrush"};

public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    MaskRCNNTorchScript(const rclcpp::NodeOptions &options);

    /**
     * Load model from TorchScript file.
     *
     * @return True if successful, false otherwise.
     */
    bool prepare() override;

    /**
     * Preprocess images for inference.
     *
     * @param images Vector of images to preprocess.
     *
     * @return True if successful, false otherwise.
     */
    bool preprocess(std::vector<sensor_msgs::msg::Image> &images) override;

    /**
     * Run inference on preprocessed images.
     *
     * @return True if successful, false otherwise.
     */
    bool predict() override;

    /**
     * Postprocess inference results.
     *
     * @return Vector of instance segmentation results.
     */
    std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> postprocess() override;

    /**
     * Cleanup allocated model resources.
     */
    void cleanup() override;
};

} // namespace cvnode_base
