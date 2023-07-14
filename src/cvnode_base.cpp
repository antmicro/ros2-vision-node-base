#include "cvnode_base/cvnode_base.hpp"

namespace cvnode_base
{

void BaseCVNode::register_callback(const rclcpp::Client<cvnode_msgs::srv::ManageCVNode>::SharedFuture future)
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
    manage_client = create_client<cvnode_msgs::srv::ManageCVNode>(manage_node_name);

    // Check if the service is available
    if (!manage_client->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "The node management service is not available after waiting");
        return;
    }

    // Create a request
    auto request = std::make_shared<cvnode_msgs::srv::ManageCVNode::Request>();
    request->type = request->REGISTER;
    request->node_name = std::string(get_name());
    request->srv_name = std::string(get_name()) + "/communication";

    // Create communication service
    communication_service = create_service<cvnode_msgs::srv::RuntimeProtocolSrv>(
        request->srv_name,
        std::bind(&BaseCVNode::communication_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Send the request
    manage_client->async_send_request(
        request,
        [this](const rclcpp::Client<cvnode_msgs::srv::ManageCVNode>::SharedFuture future)
        { register_callback(future); });
}

void BaseCVNode::unregisterNode()
{
    if (!manage_client)
    {
        RCLCPP_INFO(get_logger(), "The node is not registered");
        return;
    }

    auto request = std::make_shared<cvnode_msgs::srv::ManageCVNode::Request>();
    request->type = request->UNREGISTER;
    request->node_name = std::string(get_name());
    manage_client->async_send_request(request);
    manage_client.reset();
}

} // namespace cvnode_base
