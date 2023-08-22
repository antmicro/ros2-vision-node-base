// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <mutex>
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
     * Unregister node using the node management service.
     */
    void unregisterNode();

    /// Client to manage the BaseCVNode.
    rclcpp::Client<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_client;

    /// Communication service.
    rclcpp::Service<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedPtr communication_service;

    std::vector<sensor_msgs::msg::Image> input_data; ///< Input data

    /// Post-processed inference output
    std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> output_data;

    std::mutex data_mutex;    ///< Mutex for data access
    std::mutex process_mutex; ///< Mutex for processing access

    uint64_t process_request_id = 0; ///< ID incremented for each request

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
     *
     * @return True if preprocessing was successful, false otherwise.
     */
    virtual bool preprocess(std::vector<sensor_msgs::msg::Image> &images) = 0;

    /**
     * Run inference.
     * This function is called after preprocess stage.
     *
     * @return True if inference was successful, false otherwise.
     */
    virtual bool predict() = 0;

    /**
     * Post-process inference results.
     *
     * @return Vector of segmentation results.
     */
    virtual std::vector<kenning_computer_vision_msgs::msg::SegmentationMsg> postprocess() = 0;

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
