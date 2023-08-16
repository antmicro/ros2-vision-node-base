// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cv_bridge/cv_bridge.h>
#include <cvnode_base/utils/utils.hpp>
#include <rclcpp/rclcpp.hpp>

namespace cvnode_base
{

cv::Mat imageToMat(const sensor_msgs::msg::Image &img, const std::string &encoding)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img.encoding);
    if (img.encoding == sensor_msgs::image_encodings::TYPE_8UC3)
    {
        cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
    }
    else if (img.encoding == sensor_msgs::image_encodings::TYPE_8UC4)
    {
        cv_ptr->encoding = sensor_msgs::image_encodings::BGRA8;
    }
    else
    {
        cv_ptr->encoding = img.encoding;
    }

    if (img.encoding == encoding)
    {
        return cv_ptr->image;
    }

    try
    {
        cv_ptr = cv_bridge::cvtColor(cv_ptr, encoding);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("cvnode_base"), "cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
    return cv_ptr->image;
}

} // namespace cvnode_base
