// Copyright 2022-2024 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "cvnode_base/cvnode_base.hpp"

namespace cvnode_base
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;

BaseCVNode::BaseCVNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options)
{
    register_node("cvnode_register");
}

void BaseCVNode::prepare_callback(
    [[maybe_unused]] const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
{
    RCLCPP_DEBUG(get_logger(), "Received request to prepare the node");
    if (!prepare())
    {
        RCLCPP_ERROR(get_logger(), "Error while preparing the node");
        response->success = false;
        cleanup();
        unregister_node();
        return;
    }
    response->success = true;
    RCLCPP_DEBUG(get_logger(), "Prepared the node");
    return;
}

void BaseCVNode::process_callback(
    const SegmentCVNodeSrv::Request::SharedPtr request,
    SegmentCVNodeSrv::Response::SharedPtr response)
{
    RCLCPP_DEBUG(get_logger(), "Received request to process input data");
    response->segmentation = run_inference(request->input);
    response->success = true;
    RCLCPP_DEBUG(get_logger(), "Processed the input data");
}

void BaseCVNode::cleanup_callback(
    [[maybe_unused]] const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
{
    RCLCPP_DEBUG(get_logger(), "Cleaning up the node");
    cleanup();
    response->success = true;
    return;
}

void BaseCVNode::register_callback(const rclcpp::Client<ManageCVNode>::SharedFuture future)
{
    auto result = future.get();
    RCLCPP_DEBUG(get_logger(), "Received response from the register node service");
    if (!result)
    {
        RCLCPP_ERROR(get_logger(), "No response from the register node service");
        return;
    }
    if (!result->status)
    {
        RCLCPP_ERROR(get_logger(), "The node is not registered");
        RCLCPP_ERROR(get_logger(), "Error message: %s", result->message.c_str());
        cleanup();
        manage_client.reset();
        prepare_service.reset();
        process_service.reset();
        cleanup_service.reset();
        return;
    }

    RCLCPP_DEBUG(get_logger(), "The node was registered: %s", result->message.c_str());
}

void BaseCVNode::register_node(const std::string &manage_node_name)
{
    RCLCPP_DEBUG(get_logger(), "Registering the node");
    if (manage_client)
    {
        unregister_node();
    }

    manage_client = create_client<ManageCVNode>(manage_node_name);
    if (!manage_client->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "The node management service is not available after waiting");
        return;
    }

    auto request = std::make_shared<ManageCVNode::Request>();
    request->type = request->REGISTER;
    request->node_name = std::string(get_name());
    request->prepare_srv_name = std::string(get_name()) + "/prepare";
    request->process_srv_name = std::string(get_name()) + "/process";
    request->cleanup_srv_name = std::string(get_name()) + "/cleanup";

    prepare_service = create_service<std_srvs::srv::Trigger>(
        request->prepare_srv_name,
        std::bind(&BaseCVNode::prepare_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Create QOS with keep last 1 to avoid queueing messages
    rmw_qos_profile_t qos = rmw_qos_profile_default;
    qos.depth = 1;

    process_service = create_service<SegmentCVNodeSrv>(
        request->process_srv_name,
        std::bind(&BaseCVNode::process_callback, this, std::placeholders::_1, std::placeholders::_2),
        qos);
    cleanup_service = create_service<std_srvs::srv::Trigger>(
        request->cleanup_srv_name,
        std::bind(&BaseCVNode::cleanup_callback, this, std::placeholders::_1, std::placeholders::_2));
    manage_client->async_send_request(
        request,
        [this](const rclcpp::Client<ManageCVNode>::SharedFuture future) { register_callback(future); });
}

void BaseCVNode::unregister_node()
{
    RCLCPP_DEBUG(get_logger(), "Unregistering the node");
    if (!manage_client)
    {
        RCLCPP_DEBUG(get_logger(), "The node is not registered");
        return;
    }

    auto request = std::make_shared<ManageCVNode::Request>();
    request->type = request->UNREGISTER;
    request->node_name = std::string(get_name());
    manage_client->async_send_request(request);
    manage_client.reset();
    prepare_service.reset();
    process_service.reset();
    cleanup_service.reset();
    RCLCPP_DEBUG(get_logger(), "The node was unregistered");
}

} // namespace cvnode_base
