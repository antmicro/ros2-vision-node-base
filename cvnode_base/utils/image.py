# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Contains utility functions for Image messages manipulation."""

import cv_bridge
import numpy as np
import rclpy
from sensor_msgs.msg import Image


def imageToMat(image: Image, encoding: str) -> np.ndarray:
    """
    Convert ROS Image to Mat with specified encoding.

    Parameters
    ----------
    image : Image
        ROS Image to convert.
    encoding : str
        Encoding of the image.

    Returns
    -------
    np.ndarray
        Mat of the image.
    """
    if image.encoding == "8UC1":
        image.encoding = "mono8"
    elif image.encoding == "8UC3":
        image.encoding = "bgr8"
    elif image.encoding == "8UC4":
        image.encoding = "bgra8"

    try:
        bridge = cv_bridge.CvBridge()
        cv_ptr = bridge.imgmsg_to_cv2(image, encoding)
    except cv_bridge.CvBridgeError as e:
        rclpy.logging.get_logger("cvnode_base").error(e)
        return np.array([])
    return cv_ptr
