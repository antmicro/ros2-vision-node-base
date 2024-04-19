# ROS2 CVNode

Copyright (c) 2022-2024 [Antmicro](https://www.antmicro.com)

`CVNode` is a ROS2 node designed to facilitate integration of computer vision algorithms into inference testing infrastructure.

## Overview

`CVNode` introduces the `CVNodeBase` class, which serves as a foundational building block for creating computer vision nodes within ROS2 projects.
This base class provides essential functionality for running ROS2 nodes.
It only requires implementation of the following abstract methods to set up your computer vision algorithm:
* `prepare` - responsible for preparing the computer vision algorithm for inference (e.g. load the model).
* `run_inference` - responsible for running the computer vision algorithm on a vector of input images.
* `cleanup` - responsible for cleaning up the computer vision algorithm after inference.

`CVNode` offers both C++ and Python implementations of the `CVNodeBase` class, enabling choice of computer vision algorithm development language.

## Building CVNode

Project dependencies:
* [ROS2 Humble](https://docs.ros.org/en/humble/index.html)
* [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [cv_bridge](https://github.com/ros-perception/vision_opencv.git)
* [OpenCV](https://opencv.org/)

The `CVNodeBase` class is located in `basecvnode` target, which is a shared library.
To build the `basecvnode` target, run the following command from the root of your ROS2 workspace:
```bash
colcon build --packages-select cvnode_base
```

This will build `libbasecvnode.so`, which can later be used as a dependency for your computer vision node.

For a usage sample of the `CVNodeBase` class, see the `MaskRCNN` demo in the [examples/mask_rcnn](./examples/mask_rcnn/) directory.
You can also explore the source code of implemented nodes in the [cvnode_base/nodes](./cvnode_base/nodes) (Python implementations) and [include/cvnode_base/nodes](./include/cvnode_base/nodes) (C++ implementations) directories.
