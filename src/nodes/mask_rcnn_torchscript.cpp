// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_base/cvnode_base.hpp>
#include <cvnode_base/nodes/mask_rcnn_torchscript.hpp>
#include <cvnode_base/utils/utils.hpp>
#include <kenning_computer_vision_msgs/msg/box_msg.hpp>
#include <kenning_computer_vision_msgs/msg/mask_msg.hpp>

#include <c10/cuda/CUDACachingAllocator.h>
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
    param_descriptor.description = "Path to the file with class names";
    declare_parameter<std::string>("class_names_path", "", param_descriptor);
}

bool MaskRCNNTorchScript::prepare()
{
    std::string model_path = get_parameter("model_path").as_string();
    if (model_path.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No script path provided");
        return false;
    }

    std::string class_names_path = get_parameter("class_names_path").as_string();
    if (class_names_path.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No class names path provided");
        return false;
    }
    std::ifstream class_names_file(class_names_path);
    if (!class_names_file.good())
    {
        RCLCPP_ERROR(this->get_logger(), "Class names file does not exist: %s", class_names_path.c_str());
        return false;
    }
    std::string line;
    std::getline(class_names_file, line);
    while (std::getline(class_names_file, line))
    {
        class_names.push_back(line.substr(0, line.find(',')));
    }
    if (class_names.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No class names found in file: %s", class_names_path.c_str());
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

std::vector<SegmentationMsg> MaskRCNNTorchScript::run_inference(std::vector<sensor_msgs::msg::Image> &X)
{
    std::vector<SegmentationMsg> result;
    for (auto &frame : X)
    {
        c10::IValue input = preprocess(frame);
        MaskRCNNOutputs prediction = predict(input);
        result.push_back(postprocess(prediction, frame));
        if (device.is_cuda())
        {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }
    return result;
}

c10::IValue MaskRCNNTorchScript::preprocess(sensor_msgs::msg::Image &frame)
{
    cv::Mat cv_image = imageToMat(frame, "bgr8");
    torch::Tensor tensor_image = torch::from_blob(cv_image.data, {cv_image.rows, cv_image.cols, 3}, torch::kUInt8);
    tensor_image = tensor_image.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
    return tensor_image;
}

MaskRCNNOutputs MaskRCNNTorchScript::predict(c10::IValue &input)
{
    torch::jit::IValue output = model.forward({input});
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
    return model_output;
}

SegmentationMsg MaskRCNNTorchScript::postprocess(MaskRCNNOutputs &prediction, sensor_msgs::msg::Image &frame)
{
    using MaskMsg = kenning_computer_vision_msgs::msg::MaskMsg;
    using BoxMsg = kenning_computer_vision_msgs::msg::BoxMsg;
    SegmentationMsg msg;
    msg.frame = frame;
    std::transform(
        prediction.classes.data_ptr<int64_t>(),
        prediction.classes.data_ptr<int64_t>() + prediction.classes.numel(),
        std::back_inserter(msg.classes),
        [this](int64_t class_id) { return this->class_names.at(class_id); });
    msg.scores = std::vector<float>(
        prediction.scores.data_ptr<float>(),
        prediction.scores.data_ptr<float>() + prediction.scores.numel());
    for (int64_t j = 0; j < prediction.masks.size(0); j++)
    {
        MaskMsg mask_msg;
        cv::Mat mask =
            paste_mask(prediction.masks.select(0, j), prediction.boxes.select(0, j), frame.height, frame.width);
        mask_msg.dimension.push_back(mask.rows);
        mask_msg.dimension.push_back(mask.cols);
        mask_msg.data = std::vector<uint8_t>(mask.data, mask.data + mask.total());
        msg.masks.push_back(mask_msg);
    }
    const c10::Scalar width = c10::Scalar(static_cast<float>(frame.width));
    const c10::Scalar height = c10::Scalar(static_cast<float>(frame.height));
    prediction.boxes.select(1, 0).div_(width);
    prediction.boxes.select(1, 1).div_(height);
    prediction.boxes.select(1, 2).div_(width);
    prediction.boxes.select(1, 3).div_(height);
    prediction.boxes = prediction.boxes.to(torch::kCPU);
    for (int64_t j = 0; j < prediction.boxes.size(0); j++)
    {
        BoxMsg box;
        box.xmin = prediction.boxes.select(0, j).select(0, 0).item<float>();
        box.ymin = prediction.boxes.select(0, j).select(0, 1).item<float>();
        box.xmax = prediction.boxes.select(0, j).select(0, 2).item<float>();
        box.ymax = prediction.boxes.select(0, j).select(0, 3).item<float>();
        msg.boxes.push_back(box);
    }
    return msg;
}

void MaskRCNNTorchScript::cleanup()
{
    model = torch::jit::script::Module();
    device = torch::Device(torch::kCPU);
}

cv::Mat
MaskRCNNTorchScript::paste_mask(const torch::Tensor &mask, const torch::Tensor &box, const int height, const int width)
{
    torch::Tensor box_int = box.to(torch::kInt32);
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
