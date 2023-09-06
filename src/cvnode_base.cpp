// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "cvnode_base/cvnode_base.hpp"
#include <thread>

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
    if (!prepare())
    {
        response->success = false;
        cleanup();
        unregister_node();
        return;
    }
    response->success = true;
    return;
}

void BaseCVNode::process_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const SegmentCVNodeSrv::Request::SharedPtr request)
{
    std::thread(
        [this, header, request]()
        {
            SegmentCVNodeSrv::Response response = SegmentCVNodeSrv::Response();
            response.output = run_inference(request->input);
            response.success = true;
            process_service->send_response(*header, response);
        })
        .detach();
}

void BaseCVNode::cleanup_callback(
    [[maybe_unused]] const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
{
    cleanup();
    response->success = true;
    return;
}

void BaseCVNode::register_callback(const rclcpp::Client<ManageCVNode>::SharedFuture future)
{

    auto result = future.get();
    if (!result)
    {
        RCLCPP_ERROR(get_logger(), "[REGISTER] No response from the register node service");
        return;
    }
    if (!result->status)
    {
        RCLCPP_ERROR(get_logger(), "[REGISTER] The node is not registered");
        RCLCPP_ERROR(get_logger(), "[REGISTER] Error message: %s", result->message.c_str());
        cleanup();
        manage_client.reset();
        prepare_service.reset();
        process_service.reset();
        cleanup_service.reset();
        return;
    }

    RCLCPP_DEBUG(get_logger(), "[REGISTER] The node was registered: %s", result->message.c_str());
}

void BaseCVNode::register_node(const std::string &manage_node_name)
{
    if (manage_client)
    {
        unregister_node();
    }

    manage_client = create_client<ManageCVNode>(manage_node_name);
    if (!manage_client->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "[REGISTER] The node management service is not available after waiting");
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
    process_service = create_service<SegmentCVNodeSrv>(
        request->process_srv_name,
        std::bind(&BaseCVNode::process_callback, this, std::placeholders::_1, std::placeholders::_2));
    cleanup_service = create_service<std_srvs::srv::Trigger>(
        request->cleanup_srv_name,
        std::bind(&BaseCVNode::cleanup_callback, this, std::placeholders::_1, std::placeholders::_2));
    manage_client->async_send_request(
        request,
        [this](const rclcpp::Client<ManageCVNode>::SharedFuture future) { register_callback(future); });
}

void BaseCVNode::unregister_node()
{
    if (!manage_client)
    {
        RCLCPP_DEBUG(get_logger(), "[UNREGISTER] The node is not registered");
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
}

} // namespace cvnode_base
