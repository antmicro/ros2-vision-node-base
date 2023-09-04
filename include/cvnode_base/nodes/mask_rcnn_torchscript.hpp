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

    /// Class names
    std::vector<std::string> class_names;

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
     * Preprocess images for inference.
     *
     * @param images Vector of images to preprocess.
     *
     * @return Preprocessed input data.
     */
    std::vector<c10::IValue> preprocess(std::vector<sensor_msgs::msg::Image> &images);

    /**
     * Run inference on preprocessed images.
     *
     * @param inputs Vector of preprocessed input data.
     *
     * @return Vector of inference outputs.
     */
    std::vector<MaskRCNNOutputs> predict(std::vector<c10::IValue> &inputs);

    /**
     * Postprocess inference results.
     *
     * @param predictions Vector of inference outputs.
     * @param images Vector of input images.
     *
     * @return Vector of instance segmentation results.
     */
    std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg>
    postprocess(std::vector<MaskRCNNOutputs> &predictions, std::vector<sensor_msgs::msg::Image> &images);

    /**
     * Cleanup allocated model resources.
     */
    void cleanup() override;
};

} // namespace cvnode_base
