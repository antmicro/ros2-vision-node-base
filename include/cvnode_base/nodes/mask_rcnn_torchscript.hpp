#pragma once

#include <cvnode_base/cvnode_base.hpp>
#include <cvnode_msgs/msg/segmentation_msg.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

namespace cvnode_base
{

struct MaskRCNNOutputs
{
    at::Tensor boxes;   ///< Bounding boxes
    at::Tensor classes; ///< Class IDs
    at::Tensor masks;   ///< Masks
    at::Tensor scores;  ///< Scores

    /**
     * Get the number of instances detected.
     *
     * @return Number of instances.
     */
    int num_instances() const { return boxes.sizes()[0]; }
};

class MaskRCNNTorchScript : public BaseCVNode
{
private:
    std::string script_path;                                ///< Path to TorchScript file
    torch::jit::script::Module model;                       ///< TorchScript model
    c10::Device device = c10::Device(c10::DeviceType::CPU); ///< Device to run inference on

    std::vector<sensor_msgs::msg::Image::SharedPtr> frames; ///< Input images
    std::vector<c10::IValue> inputs;                        ///< Preprocessed input images
    std::vector<MaskRCNNOutputs> predictions;               ///< Inference outputs

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
     * @param node_name Name of the node.
     * @param script_path Path to TorchScript file.
     * @param options Node options.
     */
    MaskRCNNTorchScript(
        const std::string &node_name,
        const std::string &script_path,
        const rclcpp::NodeOptions &options)
        : BaseCVNode(node_name, options), script_path(script_path)
    {
    }

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
     */
    void preprocess(std::vector<sensor_msgs::msg::Image::SharedPtr> &images) override;

    /**
     * Run inference on the images.
     * This function is called after the images have been preprocessed.
     */
    void predict() override;

    /**
     * Postprocess the inference results.
     *
     * @return Vector of instance segmentation results.
     */
    std::vector<cvnode_msgs::msg::SegmentationMsg> postprocess() override;

    /**
     * Cleanup allocated model resources.
     */
    void cleanup() override;
};

} // namespace cvnode_base
