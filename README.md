# ROS2 CVNode

Copyright (c) 2022-2023 [Antmicro](https://www.antmicro.com)

`CVNode` is a ROS2 node designed to facilitate the integration of computer vision algorithms into an inference testing infrastructure.

## Overview

`CVNode` introduces the `CVNodeBase` class, which serves as the foundational building block for creating computer vision nodes within your ROS2 project.
This base class provides essential functionality for running a ROS2 node and only requires the implementation of a few abstract methods to get your computer vision algorithm up and running:
* `prepare` - Method responsible for preparing the computer vision algorithm for inference.
* `run_inference` - Method responsible for running the computer vision algorithm on a vector of input images.
* `cleanup` - Method responsible for cleaning up the computer vision algorithm after inference.

The `CVNode` offers both C++ and Python implementations of the `CVNodeBase` class, allowing choice in the language used to develop computer vision algorithm.

## Building the CVNode

Project dependencies:
* [ROS2 Humble](https://docs.ros.org/en/humble/index.html)
* [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [cv_bridge](https://github.com/ros-perception/vision_opencv.git)
* `OpenCV`

To build the `CVNodeBase` target, run the following command from the root of your ROS2 workspace:
```bash
colcon build --packages-select cvnode_base
```

This will build the `CVNodeBase` target, which can be used as a dependency for your computer vision node.

For example usage of the `CVNodeBase` class, see the `MaskRCNN` demo in [examples/mask_rcnn](examples/mask_rcnn/) directory.
Also, source code of already implemented nodes can be explored in the [cvnode_base/nodes](cvnode_base/nodes) directory for Python implementations and [include/cvnode_base/nodes](include/cvnode_base/nodes) directory for C++ ones.
