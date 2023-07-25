#include <cvnode_base/nodes/mask_rcnn_torchscript.hpp>
#include <cvnode_base/utils/utils.hpp>
#include <cvnode_msgs/msg/box_msg.hpp>
#include <cvnode_msgs/msg/mask_msg.hpp>
#include <opencv2/opencv.hpp>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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
            tuple_outputs[2].toTensor().to(torch::kCPU),
            tuple_outputs[3].toTensor().to(torch::kCPU)};
        predictions.push_back(model_output);
    }
}

std::vector<cvnode_msgs::msg::SegmentationMsg> MaskRCNNTorchScript::postprocess()
{
    std::vector<cvnode_msgs::msg::SegmentationMsg> segmentations;
    cvnode_msgs::msg::SegmentationMsg msg;
    MaskRCNNOutputs output;
    at::Tensor masks;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        output = predictions.at(i);
        masks = ((output.masks > 0.6).to(torch::kUInt8) * 255).contiguous();
        const c10::Scalar width = c10::Scalar(static_cast<float>(frames.at(i)->width));
        const c10::Scalar height = c10::Scalar(static_cast<float>(frames.at(i)->height));
        output.boxes.select(1, 0).div_(width);
        output.boxes.select(1, 1).div_(height);
        output.boxes.select(1, 2).div_(width);
        output.boxes.select(1, 3).div_(height);
        msg.frame = *frames.at(i).get();
        std::transform(
            output.classes.data_ptr<int64_t>(),
            output.classes.data_ptr<int64_t>() + output.classes.numel(),
            std::back_inserter(msg.classes),
            [this](int64_t class_id) { return this->class_names.at(class_id); });
        msg.scores = std::vector<float>(
            output.scores.data_ptr<float>(),
            output.scores.data_ptr<float>() + output.scores.numel());
        for (int64_t j = 0; j < output.boxes.size(0); j++)
        {
            cvnode_msgs::msg::BoxMsg box;
            box.xmin = output.boxes.select(0, j).select(0, 0).item<float>();
            box.ymin = output.boxes.select(0, j).select(0, 1).item<float>();
            box.xmax = output.boxes.select(0, j).select(0, 2).item<float>();
            box.ymax = output.boxes.select(0, j).select(0, 3).item<float>();
            msg.boxes.push_back(box);
        }
        for (size_t j = 0; j < output.masks.size(0); j++)
        {
            cvnode_msgs::msg::MaskMsg mask;
            mask.dimension = std::vector<int64_t>{output.masks.select(0, j).size(1), output.masks.select(0, j).size(2)};
            std::transform(
                output.masks.select(0, j).data_ptr<float>(),
                output.masks.select(0, j).data_ptr<float>() + output.masks.select(0, j).numel(),
                std::back_inserter(mask.data),
                [](float f) { return static_cast<uint8_t>(f * 255); });
            msg.masks.push_back(mask);
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

} // namespace cvnode_base
