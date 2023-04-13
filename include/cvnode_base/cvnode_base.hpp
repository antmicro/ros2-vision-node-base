#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <string>

#include <cvnode_msgs/srv/register_cv_node.hpp>
#include <cvnode_msgs/srv/unregister_cv_node.hpp>

namespace cvnode_base
{

/**
 * Base class for all computer vision nodes managed by the manager class.
 *
 * @tparam T Type of the process service.
 */
template <class T> class BaseCVNode : public rclcpp::Node
{
private:
    /**
     * Callback for the registration service.
     * If the node was registered successfuly, the unregister service is created.
     *
     * @param unregister_service_name Name of the unregister service.
     * @param future Future of the service.
     */
    void register_callback(const std::string &unregister_service_name,
                           const rclcpp::Client<cvnode_msgs::srv::RegisterCVNode>::SharedFuture future)
    {

        auto result = future.get();
        if (!result)
        {
            RCLCPP_ERROR(this->get_logger(), "No response from the register node service");
            return;
        }
        if (!result->status)
        {
            RCLCPP_ERROR(this->get_logger(), "The node is not registered");
            RCLCPP_ERROR(this->get_logger(), "Error message: %s", result->message.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "%s", result->message.c_str());
        this->unregister_client = this->create_client<cvnode_msgs::srv::UnregisterCVNode>(unregister_service_name);
    }

    /**
     * Callback for the algorithm prepare service.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    virtual void prepare_callback(const std_srvs::srv::Trigger::Request::SharedPtr request,
                                  std_srvs::srv::Trigger::Response::SharedPtr response) = 0;

    /**
     * Callback for the algorithm process service.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    virtual void process_callback(const typename T::Request::SharedPtr request,
                                  typename T::Response::SharedPtr response) = 0;
    /**
     * Callback for the algorithm cleanup service.
     *
     * @param request Request of the service.
     * @param response Response of the service.
     */
    virtual void cleanup_callback(const std_srvs::srv::Trigger::Request::SharedPtr request,
                                  std_srvs::srv::Trigger::Response::SharedPtr response) = 0;

    /**
     * Unregisters the node using the unregister service.
     */
    void unregisterNode()
    {
        auto request = std::make_shared<cvnode_msgs::srv::UnregisterCVNode::Request>();
        request->node_name = std::string(get_name());
        unregister_client->async_send_request(request);
        register_client.reset();
        unregister_client.reset();
    }

    /// Client to register a new BaseCVNode.
    rclcpp::Client<cvnode_msgs::srv::RegisterCVNode>::SharedPtr register_client;

    /// Client to unregister a BaseCVNode.
    rclcpp::Client<cvnode_msgs::srv::UnregisterCVNode>::SharedPtr unregister_client;

    /// Service to prepare the algorithm and allocate resources.
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr prepare_service;

    /// Service to process the image through the algorithm.
    typename rclcpp::Service<T>::SharedPtr process_service;

    /// Service to deallocate resources.
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cleanup_service;

public:
    /**
     * Constructor.
     *
     * @param node_name Name of the node.
     * @param options Node options.
     */
    BaseCVNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {}

    /**
     * Registers the node using the register service.
     *
     * @param register_service_name Name of the register service.
     * @param unregister_service_name Name of the unregister service.
     */
    void registerNode(const std::string &register_service_name, const std::string &unregister_service_name)
    {
        // Create a service client to register the node
        register_client = this->create_client<cvnode_msgs::srv::RegisterCVNode>(register_service_name);

        // Check if the service is available
        if (!register_client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_ERROR(this->get_logger(), "The registration service is not available after waiting");
            return;
        }

        // Unregister node if needed
        if (unregister_client)
        {
            unregisterNode();
        }

        // Create a request
        auto request = std::make_shared<cvnode_msgs::srv::RegisterCVNode::Request>();
        request->node_name = std::string(get_name());
        request->prepare_service_name = request->node_name + "_prepare";
        request->process_service_name = request->node_name + "_process";
        request->cleanup_service_name = request->node_name + "_cleanup";

        // Create service servers
        prepare_service = this->prepare_service = this->create_service<std_srvs::srv::Trigger>(
            request->prepare_service_name,
            std::bind(&BaseCVNode::prepare_callback, this, std::placeholders::_1, std::placeholders::_2));

        process_service = this->create_service<T>(
            request->process_service_name,
            std::bind(&BaseCVNode::process_callback, this, std::placeholders::_1, std::placeholders::_2));

        cleanup_service = this->create_service<std_srvs::srv::Trigger>(
            request->cleanup_service_name,
            std::bind(&BaseCVNode::cleanup_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Send the request
        register_client->async_send_request(
            request,
            [this, unregister_service_name](const rclcpp::Client<cvnode_msgs::srv::RegisterCVNode>::SharedFuture future)
            { register_callback(unregister_service_name, future); });
    }

    /**
     * Destructor.
     */
    ~BaseCVNode()
    {
        if (!unregister_client)
        {
            return;
        }
        unregisterNode();
    }
};

} // namespace cvnode_base
