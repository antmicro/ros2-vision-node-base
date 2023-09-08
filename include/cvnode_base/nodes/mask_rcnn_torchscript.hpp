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
    torch::Tensor boxes;   ///< Bounding boxes
    torch::Tensor classes; ///< Class IDs
    torch::Tensor masks;   ///< Masks
    torch::Tensor scores;  ///< Scores
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
    cv::Mat paste_mask(const torch::Tensor &mask, const torch::Tensor &box, const int height, const int width);

    std::string model_path;                                 ///< Path to TorchScript file
    torch::jit::script::Module model;                       ///< TorchScript model
    c10::Device device = c10::Device(c10::DeviceType::CPU); ///< Device to run inference on
    std::vector<std::string> class_names;                   ///< Vector of class names

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
     * Run inference on input images.
     *
     * @param X Vector of input images.
     *
     * @return Vector of instance segmentation results.
     */
    std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg>
    run_inference(std::vector<sensor_msgs::msg::Image> &X) override;

    /**
     * Preprocess input image for inference.
     *
     * @param frame Image to preprocess.
     *
     * @return Preprocessed input data.
     */
    c10::IValue preprocess(sensor_msgs::msg::Image &frame);

    /**
     * Run inference on preprocessed image.
     *
     * @param input Preprocessed input data.
     *
     * @return Inference outputs.
     */
    MaskRCNNOutputs predict(c10::IValue &input);

    /**
     * Postprocess inference result.
     *
     * @param prediction Inference outputs.
     * @param frame Input image.
     *
     * @return Instance segmentation result.
     */
    kenning_computer_vision_msgs::msg::SegmentationMsg
    postprocess(MaskRCNNOutputs &prediction, sensor_msgs::msg::Image &frame);

    /**
     * Cleanup allocated model resources.
     */
    void cleanup() override;
};

} // namespace cvnode_base
