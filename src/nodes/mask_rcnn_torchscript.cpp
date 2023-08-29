// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_base/cvnode_base.hpp>
#include <cvnode_base/nodes/mask_rcnn_torchscript.hpp>
#include <cvnode_base/utils/utils.hpp>
#include <kenning_computer_vision_msgs/msg/box_msg.hpp>
#include <kenning_computer_vision_msgs/msg/mask_msg.hpp>

#include <c10/cuda/CUDAStream.h>
#include <opencv2/opencv.hpp>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/nn/functional/vision.h>

namespace cvnode_base
{

using SegmentationMsg = kenning_computer_vision_msgs::msg::SegmentationMsg;

MaskRCNNTorchScript::MaskRCNNTorchScript(const rclcpp::NodeOptions &options)
    : BaseCVNode("mask_rcnn_torchscript_node", options)
{
    rcl_interfaces::msg::ParameterDescriptor param_descriptor;
    param_descriptor.read_only = false;
    param_descriptor.description = "Path to the TorchScript model file";
    declare_parameter<std::string>("model_path", "", param_descriptor);
}

bool MaskRCNNTorchScript::prepare()
{
    std::string model_path = get_parameter("model_path").as_string();
    if (model_path.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No script path provided");
        return false;
    }

    torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
    torch::jit::setFusionStrategy(strategy);
    torch::autograd::AutoGradMode guard(false);
    model = torch::jit::load(model_path);
    if (model.buffers().size() == 0)
    {
        RCLCPP_ERROR(this->get_logger(), "No parameters found in script: %s", model_path.c_str());
        return false;
    }
    device = (*std::begin(model.buffers())).device();
    return true;
}

std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg>
MaskRCNNTorchScript::run_inference(std::vector<sensor_msgs::msg::Image> &X)
{
    std::vector<c10::IValue> inputs = preprocess(X);
    std::vector<MaskRCNNOutputs> outputs = predict(inputs);
    inputs.clear();
    return postprocess(outputs, X);
}

std::vector<c10::IValue> MaskRCNNTorchScript::preprocess(std::vector<sensor_msgs::msg::Image> &images)
{
    std::vector<c10::IValue> inputs;
    for (auto &frame : images)
    {
        cv::Mat cv_image = imageToMat(frame, "bgr8");
        torch::Tensor tensor_image = torch::from_blob(cv_image.data, {cv_image.rows, cv_image.cols, 3}, torch::kUInt8);
        tensor_image = tensor_image.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
        inputs.push_back(tensor_image);
    }
    return inputs;
}

std::vector<MaskRCNNOutputs> MaskRCNNTorchScript::predict(std::vector<c10::IValue> &inputs)
{
    std::vector<MaskRCNNOutputs> predictions;
    for (auto &input : inputs)
    {
        c10::IValue output = model.forward({input});
        if (device.is_cuda())
        {
            c10::cuda::getCurrentCUDAStream().synchronize();
        }
        auto tuple_outputs = output.toTuple()->elements();
        MaskRCNNOutputs model_output{
            tuple_outputs[0].toTensor().to(torch::kCPU),
            tuple_outputs[1].toTensor().to(torch::kCPU),
            tuple_outputs[2].toTensor().squeeze(1).to(torch::kCPU),
            tuple_outputs[3].toTensor().to(torch::kCPU)};
        predictions.push_back(model_output);
    }
    return predictions;
}

std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> MaskRCNNTorchScript::postprocess(
    std::vector<MaskRCNNOutputs> &predictions,
    std::vector<sensor_msgs::msg::Image> &images)
{
    using MaskMsg = kenning_computer_vision_msgs::msg::MaskMsg;
    using BoxMsg = kenning_computer_vision_msgs::msg::BoxMsg;
    std::vector<SegmentationMsg> segmentations;
    SegmentationMsg msg;
    MaskRCNNOutputs output;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        output = predictions.at(i);
        msg.frame = images.at(i);
        std::transform(
            output.classes.data_ptr<int64_t>(),
            output.classes.data_ptr<int64_t>() + output.classes.numel(),
            std::back_inserter(msg.classes),
            [this](int64_t class_id) { return this->class_names.at(class_id); });
        msg.scores = std::vector<float>(
            output.scores.data_ptr<float>(),
            output.scores.data_ptr<float>() + output.scores.numel());
        for (int64_t j = 0; j < output.masks.size(0); j++)
        {
            MaskMsg mask_msg;
            cv::Mat mask = paste_mask(
                output.masks.select(0, j),
                output.boxes.select(0, j),
                images.at(i).height,
                images.at(i).width);
            mask_msg.dimension.push_back(mask.rows);
            mask_msg.dimension.push_back(mask.cols);
            mask_msg.data = std::vector<uint8_t>(mask.data, mask.data + mask.total());
            msg.masks.push_back(mask_msg);
        }
        const c10::Scalar width = c10::Scalar(static_cast<float>(images.at(i).width));
        const c10::Scalar height = c10::Scalar(static_cast<float>(images.at(i).height));
        output.boxes.select(1, 0).div_(width);
        output.boxes.select(1, 1).div_(height);
        output.boxes.select(1, 2).div_(width);
        output.boxes.select(1, 3).div_(height);
        output.boxes = output.boxes.to(torch::kCPU);
        for (int64_t j = 0; j < output.boxes.size(0); j++)
        {
            BoxMsg box;
            box.xmin = output.boxes.select(0, j).select(0, 0).item<float>();
            box.ymin = output.boxes.select(0, j).select(0, 1).item<float>();
            box.xmax = output.boxes.select(0, j).select(0, 2).item<float>();
            box.ymax = output.boxes.select(0, j).select(0, 3).item<float>();
            msg.boxes.push_back(box);
        }
        segmentations.push_back(msg);
    }
    return segmentations;
}

void MaskRCNNTorchScript::cleanup()
{
    model = torch::jit::script::Module();
    device = torch::Device(torch::kCPU);
}

cv::Mat
MaskRCNNTorchScript::paste_mask(const at::Tensor &mask, const at::Tensor &box, const int height, const int width)
{
    at::Tensor box_int = box.to(torch::kInt32);
    int32_t xmin = std::max(box_int.select(0, 0).item<int32_t>(), 0);
    int32_t ymin = std::max(box_int.select(0, 1).item<int32_t>(), 0);
    int32_t xmax = std::min(box_int.select(0, 2).item<int32_t>(), width);
    int32_t ymax = std::min(box_int.select(0, 3).item<int32_t>(), height);

    int32_t x_size = xmax - xmin;
    int32_t y_size = ymax - ymin;

    if (x_size <= 0 || y_size <= 0)
    {
        return cv::Mat::zeros(height, width, CV_8UC1);
    }

    cv::Mat mask_mat = cv::Mat(mask.size(0), mask.size(1), CV_32FC1, mask.data_ptr<float>());
    cv::resize(mask_mat, mask_mat, cv::Size(x_size, y_size), cv::INTER_LINEAR);
    cv::threshold(mask_mat, mask_mat, 0.5, 255, cv::THRESH_BINARY);

    cv::Mat img_mat = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat roi = img_mat(cv::Rect(xmin, ymin, x_size, y_size));
    cv::bitwise_or(roi, mask_mat, roi);
    img_mat.convertTo(img_mat, CV_8UC1, 255);
    return img_mat;
}

} // namespace cvnode_base

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_base::MaskRCNNTorchScript)
