// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <kenning_computer_vision_msgs/srv/manage_cv_node.hpp>
#include <kenning_computer_vision_msgs/srv/segment_cv_node_srv.hpp>

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
    void register_callback(const rclcpp::Client<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedFuture future);

    /**
     * Callback for communication service.
     * Responsible for handling the communication between the manager and the node.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void communication_callback(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr request);

    /**
     * Reports error to the manager.
     *
     * @param header Header of the service request.
     * @param error_msg Error message.
     */
    void report_error(const std::shared_ptr<rmw_request_id_t> header, const std::string &error_msg);

    /**
     * Unregister node using the node management service.
     */
    void unregister_node();

    /// Client to manage the BaseCVNode.
    rclcpp::Client<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_client;

    /// Communication service.
    rclcpp::Service<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedPtr communication_service;

public:
    /**
     * Constructor.
     *
     * @param node_name Name of the node.
     * @param options Node options.
     */
    BaseCVNode(const std::string &node_name, const rclcpp::NodeOptions &options);

    /**
     * Register node using the node management service.
     *
     * @param manage_node_name Name of the service to manage the node.
     */
    void register_node(const std::string &manage_node_name);

    /**
     * Prepare node and model for inference.
     *
     * @return True if successful, false otherwise.
     */
    virtual bool prepare() = 0;

    /**
     * Run inference on the input data.
     *
     * @param X Input data.
     *
     * @return Inference output.
     */
    virtual std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg>
    run_inference(std::vector<sensor_msgs::msg::Image> &X) = 0;

    /**
     * Cleanup allocated model resources.
     */
    virtual void cleanup() = 0;

    /**
     * Destructor.
     */
    ~BaseCVNode() { unregister_node(); }
};

} // namespace cvnode_base
