#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include <cvnode_msgs/msg/segmentation_msg.hpp>
#include <cvnode_msgs/srv/manage_cv_node.hpp>
#include <cvnode_msgs/srv/runtime_protocol_srv.hpp>

namespace cvnode_base
{

/**
 * Base class for all computer vision nodes managed by the manager class.
 */
class BaseCVNode : public rclcpp::Node
{
private:
    /**
     * Callback for registration service.
     *
     * @param future Future of the service.
     */
    void register_callback(const rclcpp::Client<cvnode_msgs::srv::ManageCVNode>::SharedFuture future);

    /**
     * Callback for communication service.
     * Responsible for handling the communication between the manager and the node.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    virtual void communication_callback(
        const cvnode_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr request,
        cvnode_msgs::srv::RuntimeProtocolSrv::Response::SharedPtr response) = 0;

    /**
     * Unregister node using the node management service.
     */
    void unregisterNode();

    /// Client to manage the BaseCVNode.
    rclcpp::Client<cvnode_msgs::srv::ManageCVNode>::SharedPtr manage_client;

    /// Communication service.
    rclcpp::Service<cvnode_msgs::srv::RuntimeProtocolSrv>::SharedPtr communication_service;

public:
    /**
     * Constructor.
     *
     * @param node_name Name of the node.
     * @param options Node options.
     */
    BaseCVNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {}

    /**
     * Register node using the node management service.
     *
     * @param manage_node_name Name of the service to manage the node.
     */
    void registerNode(const std::string &manage_node_name);

    /**
     * Prepare node and model for inference.
     *
     * @return True if successful, false otherwise.
     */
    virtual bool prepare() = 0;

    /**
     * Preprocess images for inference.
     *
     * @param images Vector of images to preprocess.
     */
    virtual void preprocess(std::vector<sensor_msgs::msg::Image::SharedPtr> &images) = 0;

    /**
     * Run inference on the images.
     * This function is called after the images have been preprocessed.
     */
    virtual void predict() = 0;

    /**
     * Postprocess the inference results.
     *
     * @return Vector of instance segmentation results.
     */
    virtual std::vector<cvnode_msgs::msg::SegmentationMsg> postprocess() = 0;

    /**
     * Cleanup allocated model resources.
     */
    virtual void cleanup() = 0;

    /**
     * Destructor.
     */
    ~BaseCVNode() { unregisterNode(); }
};

} // namespace cvnode_base
