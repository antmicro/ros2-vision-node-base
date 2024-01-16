// Copyright 2022-2024 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.h>

namespace cvnode_base
{

/**
 * Convert a sensor_msgs::Image to a cv::Mat with a given encoding.
 *
 * @param img The sensor_msgs::Image to convert.
 * @param encoding The encoding to use for the cv::Mat.
 *
 * @return Converted cv::Mat.
 */
cv::Mat imageToMat(const sensor_msgs::msg::Image &img, const std::string &encoding);

} // namespace cvnode_base
