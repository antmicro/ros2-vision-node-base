// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "cvnode_base/cvnode_base.hpp"
#include <kenning_computer_vision_msgs/runtime_msg_type.hpp>

namespace cvnode_base
{

using ManageCVNode = kenning_computer_vision_msgs::srv::ManageCVNode;
using SegmentCVNodeSrv = kenning_computer_vision_msgs::srv::SegmentCVNodeSrv;

void BaseCVNode::communication_callback(
    const std::shared_ptr<rmw_request_id_t> header,
    const SegmentCVNodeSrv::Request::SharedPtr request)
{
    using namespace kenning_computer_vision_msgs::runtime_message_type;
    SegmentCVNodeSrv::Response response = SegmentCVNodeSrv::Response();
    switch (request->message_type)
    {
    case MODEL:
        if (!prepare())
        {
            response.message_type = ERROR;
            RCLCPP_ERROR(get_logger(), "Failed to prepare node. Closing.");
            communication_service->send_response(*header, response);
            cleanup();
            unregisterNode();
            break;
        }
        response.message_type = OK;
        communication_service->send_response(*header, response);
        break;
    case ERROR:
        response.message_type = ERROR;
        RCLCPP_ERROR(get_logger(), "Received ERROR message. Cleaning up.");
        communication_service->send_response(*header, response);
        cleanup();
        unregisterNode();
        break;
    case DATA:
        if (request->input.size() == 0)
        {
            RCLCPP_ERROR(get_logger(), "Received empty data");
            response.message_type = ERROR;
            communication_service->send_response(*header, response);
            break;
        }
        input_data = request->input;
        response.message_type = OK;
        communication_service->send_response(*header, response);
        break;
    case PROCESS:
        if (!preprocess(input_data))
        {
            RCLCPP_ERROR(get_logger(), "Preprocessing failed");
            response.message_type = ERROR;
            communication_service->send_response(*header, response);
            break;
        }
        if (!predict())
        {
            RCLCPP_ERROR(get_logger(), "Inference failed");
            response.message_type = ERROR;
            communication_service->send_response(*header, response);
            break;
        }
        output_data = postprocess();
        response.message_type = OK;
        communication_service->send_response(*header, response);
        input_data.clear();
        break;
    case OUTPUT:
        if (output_data.size() == 0)
        {
            RCLCPP_WARN(get_logger(), "[OUTPUT] No output data, returning empty message");
        }
        response.output = output_data;
        response.message_type = OK;
        output_data.clear();
        communication_service->send_response(*header, response);
        break;
    default:
        RCLCPP_ERROR(get_logger(), "Unknown message type. Aborting.");
        response.message_type = ERROR;
        communication_service->send_response(*header, response);
        break;
    }
}

void BaseCVNode::register_callback(const rclcpp::Client<ManageCVNode>::SharedFuture future)
{

    auto result = future.get();
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
        communication_service.reset();
        return;
    }

    RCLCPP_INFO(get_logger(), "The node was registered: %s", result->message.c_str());
}

void BaseCVNode::registerNode(const std::string &manage_node_name)
{
    // Unregister node if was registered before
    if (manage_client)
    {
        unregisterNode();
    }

    // Create a service client to manage the node
    manage_client = create_client<ManageCVNode>(manage_node_name);

    // Check if the service is available
    if (!manage_client->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "The node management service is not available after waiting");
        return;
    }

    // Create a request
    auto request = std::make_shared<ManageCVNode::Request>();
    request->type = request->REGISTER;
    request->node_name = std::string(get_name());
    request->srv_name = std::string(get_name()) + "/communication";

    // Create communication service
    communication_service = create_service<SegmentCVNodeSrv>(
        request->srv_name,
        std::bind(&BaseCVNode::communication_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Send the request
    manage_client->async_send_request(
        request,
        [this](const rclcpp::Client<ManageCVNode>::SharedFuture future) { register_callback(future); });
}

void BaseCVNode::unregisterNode()
{
    if (!manage_client)
    {
        RCLCPP_INFO(get_logger(), "The node is not registered");
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
