#include <cvnode_base/nodes/mask_rcnn_torchscript.hpp>
#include <cvnode_base/utils/utils.hpp>
#include <kenning_computer_vision_msgs/msg/box_msg.hpp>
#include <kenning_computer_vision_msgs/msg/mask_msg.hpp>
#include <opencv2/opencv.hpp>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/nn/functional/vision.h>

namespace cvnode_base
{

bool MaskRCNNTorchScript::prepare()
{
    torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
    torch::jit::setFusionStrategy(strategy);
    torch::autograd::AutoGradMode guard(false);
    model = torch::jit::load(this->script_path);
    if (model.buffers().size() == 0)
    {
        RCLCPP_ERROR(this->get_logger(), "No parameters found in script: %s", this->script_path.c_str());
        return false;
    }
    device = (*std::begin(model.buffers())).device();
    RCLCPP_INFO(this->get_logger(), "Successfuly loaded script %s", this->script_path.c_str());
    return true;
}

void MaskRCNNTorchScript::preprocess(std::vector<sensor_msgs::msg::Image::SharedPtr> &images)
{
    inputs.clear();
    frames.clear();
    frames = images;
    for (auto &frame : frames)
    {
        cv::Mat cv_image = imageToMat(frame, "bgr8");
        torch::Tensor tensor_image = torch::from_blob(cv_image.data, {cv_image.rows, cv_image.cols, 3}, torch::kUInt8);
        tensor_image = tensor_image.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
        inputs.push_back(tensor_image);
    }
}

void MaskRCNNTorchScript::predict()
{
    predictions.clear();
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
}

std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> MaskRCNNTorchScript::postprocess()
{
    std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> segmentations;
    kenning_computer_vision_msgs::msg::SegmentationMsg msg;
    MaskRCNNOutputs output;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        output = predictions.at(i);
        msg.frame = *frames.at(i).get();
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
            kenning_computer_vision_msgs::msg::MaskMsg mask_msg;
            cv::Mat mask = paste_mask(
                output.masks.select(0, j),
                output.boxes.select(0, j),
                frames.at(i)->height,
                frames.at(i)->width);
            mask_msg.dimension.push_back(mask.rows);
            mask_msg.dimension.push_back(mask.cols);
            mask.convertTo(mask, CV_8UC1, 255);
            mask_msg.data = std::vector<uint8_t>(mask.data, mask.data + mask.total());
            msg.masks.push_back(mask_msg);
        }
        const c10::Scalar width = c10::Scalar(static_cast<float>(frames.at(i)->width));
        const c10::Scalar height = c10::Scalar(static_cast<float>(frames.at(i)->height));
        output.boxes.select(1, 0).div_(width);
        output.boxes.select(1, 1).div_(height);
        output.boxes.select(1, 2).div_(width);
        output.boxes.select(1, 3).div_(height);
        output.boxes = output.boxes.to(torch::kCPU);
        for (int64_t j = 0; j < output.boxes.size(0); j++)
        {
            kenning_computer_vision_msgs::msg::BoxMsg box;
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
    inputs.clear();
    frames.clear();
    predictions.clear();
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

    cv::Mat mask_mat = cv::Mat(mask.size(0), mask.size(1), CV_32FC1, mask.data_ptr<float>());
    cv::resize(mask_mat, mask_mat, cv::Size(xmax - xmin, ymax - ymin), cv::INTER_LINEAR);
    cv::threshold(mask_mat, mask_mat, 0.5, 255, cv::THRESH_BINARY);

    cv::Mat img_mat = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat roi = img_mat(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    cv::bitwise_or(roi, mask_mat, roi);
    return img_mat;
}

} // namespace cvnode_base
