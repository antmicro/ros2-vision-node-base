// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
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
     * Callback for prepare service.
     * Prepares the node for inference.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    void prepare_callback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response);

    /**
     * Callback for process service.
     * Runs inference on the input data.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    void process_callback(
        const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr request,
        kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Response::SharedPtr response);

    /**
     * Callback for cleanup service.
     * Cleans up the node after inference.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    void cleanup_callback(
        const std_srvs::srv::Trigger::Request::SharedPtr request,
        std_srvs::srv::Trigger::Response::SharedPtr response);

    /**
     * Unregister node using the node management service.
     */
    void unregister_node();

    /// Client to manage the BaseCVNode.
    rclcpp::Client<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_client;

    /// Service to prepare the node for inference.
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr prepare_service;
    /// Service to run inference on the input data.
    rclcpp::Service<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedPtr process_service;
    /// Service to cleanup the node after inference.
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cleanup_service;

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
