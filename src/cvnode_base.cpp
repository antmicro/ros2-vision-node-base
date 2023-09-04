// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "cvnode_base/cvnode_base.hpp"
#include <kenning_computer_vision_msgs/msg/runtime_msg_type.hpp>
#include <thread>

namespace cvnode_base
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;
using RuntimeMsgType = kenning_computer_vision_msgs::msg::RuntimeMsgType;

BaseCVNode::BaseCVNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options)
{
    register_node("node_manager/register");
}

void BaseCVNode::communication_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const SegmentCVNodeSrv::Request::SharedPtr request)
{
    SegmentCVNodeSrv::Response response = SegmentCVNodeSrv::Response();
    switch (request->message_type)
    {
    case RuntimeMsgType::MODEL:
        if (!prepare())
        {
            report_error(header, "[MODEL] Failed to prepare node.");
            cleanup();
            unregister_node();
            break;
        }
        response.message_type = RuntimeMsgType::OK;
        communication_service->send_response(*header, response);
        break;
    case RuntimeMsgType::PROCESS:
        std::thread(
            [this, header, request]()
            {
                SegmentCVNodeSrv::Response response = SegmentCVNodeSrv::Response();
                response.output = run_inference(request->input);
                response.message_type = RuntimeMsgType::OK;
                communication_service->send_response(*header, response);
            })
            .detach();
        break;
    case RuntimeMsgType::CLEANUP:
        cleanup();
        response.message_type = RuntimeMsgType::OK;
        communication_service->send_response(*header, response);
        break;
    case RuntimeMsgType::ERROR:
        report_error(header, "[ERROR] Received ERROR message");
        cleanup();
        unregister_node();
        break;
    default:
        report_error(header, "[UNKNOWN] Received unknown message type");
        break;
    }
}

void BaseCVNode::report_error(const std::shared_ptr<rmw_request_id_t> header, const std::string &message)
{
    SegmentCVNodeSrv::Response response = SegmentCVNodeSrv::Response();
    response.message_type = RuntimeMsgType::ERROR;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    communication_service->send_response(*header, response);
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
        communication_service.reset();
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
    request->srv_name = std::string(get_name()) + "/communication";

    communication_service = create_service<SegmentCVNodeSrv>(
        request->srv_name,
        std::bind(&BaseCVNode::communication_callback, this, std::placeholders::_1, std::placeholders::_2));
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
    communication_service.reset();
}

} // namespace cvnode_base
