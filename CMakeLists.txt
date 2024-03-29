# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.8)
project(cvnode_base)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_YOLACT "Builds nodes for YOLACT instance segmentation model" OFF)
option(BUILD_MASK_RCNN "Builds nodes for Mask R-CNN instance segmentation model except for TorchScript implementation" OFF)
option(BUILD_MASK_RCNN_TORCHSCRIPT "Builds node with TorchScript implementation of the MaskRCNN model" OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(kenning_computer_vision_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(include)

if(BUILD_TESTING)
    find_package(ament_cmake_clang_format REQUIRED)
    find_package(ament_cmake_clang_tidy REQUIRED)

    ament_clang_format(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
        "include"
        "src"
    )

    ament_clang_tidy(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
       ${CMAKE_BINARY_DIR}/compile_commands.json
       "include"
       "src"
    )
endif()

#############

add_library(basecvnode SHARED
    src/cvnode_base.cpp
    src/utils/utils.cpp
)

ament_target_dependencies(basecvnode
    cv_bridge
    kenning_computer_vision_msgs
    rclcpp
    sensor_msgs
    std_srvs
)

target_link_libraries(basecvnode
    ${OpenCV_LIBS}
)

ament_export_targets(basecvnodeTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
    cv_bridge
    kenning_computer_vision_msgs
    rclcpp
    sensor_msgs
    std_srvs
)

if (BUILD_MASK_RCNN_TORCHSCRIPT)

    option(BUILD_TORCHVISION "Clones and builds TorchVision library" OFF)

    # Node dependencies
    find_package(rcl_interfaces REQUIRED)
    find_package(rclcpp_components REQUIRED)

    find_package(Torch REQUIRED)
    find_package(TorchVision QUIET)
    if (NOT TorchVision_FOUND AND BUILD_TORCHVISION)
        message("TorchVision not found - building it")
        include(FetchContent)
        # Fetch and build torchvision with `cmake -DWITH_CUDA=ON`
        FetchContent_Declare(
            torchvision
            GIT_REPOSITORY https://github.com/pytorch/vision
            GIT_TAG v0.15.1
        )
        set(WITH_CUDA ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(torchvision)
        add_library(TorchVision::TorchVision ALIAS torchvision)
    elseif (NOT TorchVision_FOUND)
        message(FATAL_ERROR "TorchVision not found - please install it or set BUILD_TORCHVISION to ON")
    endif()

    add_library(mask_rcnn_torchscript_node SHARED
        src/nodes/mask_rcnn_torchscript.cpp
    )

    ament_target_dependencies(mask_rcnn_torchscript_node
        rclcpp_components
        rcl_interfaces
    )

    target_link_libraries(mask_rcnn_torchscript_node
        basecvnode
        -Wl,--no-as-needed TorchVision::TorchVision
        -Wl,--as-needed ${TORCH_LIBRARIES}
    )

    rclcpp_components_register_nodes(mask_rcnn_torchscript_node "cvnode_base::MaskRCNNTorchScript")

    install(TARGETS
        mask_rcnn_torchscript_node
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )

    install(FILES
        examples/mask_rcnn/launch/mask_rcnn_torchscript_launch.py
        DESTINATION share/${PROJECT_NAME})
endif(BUILD_MASK_RCNN_TORCHSCRIPT)

if(BUILD_MASK_RCNN)
    install(PROGRAMS
        cvnode_base/nodes/mask_rcnn_detectron.py
        cvnode_base/nodes/mask_rcnn_onnx.py
        cvnode_base/nodes/mask_rcnn_tensorrt.py
        DESTINATION lib/${PROJECT_NAME})
    install(FILES
        examples/mask_rcnn/launch/mask_rcnn_detectron_launch.py
        examples/mask_rcnn/launch/mask_rcnn_onnx_launch.py
        examples/mask_rcnn/launch/mask_rcnn_tensorrt_launch.py
        DESTINATION share/${PROJECT_NAME})
endif(BUILD_MASK_RCNN)

if(BUILD_YOLACT)
    install(PROGRAMS
        cvnode_base/nodes/yolact_kenning.py
        cvnode_base/nodes/yolact_tensorrt.py
        DESTINATION lib/${PROJECT_NAME})
    install(FILES
        examples/yolact/launch/yolact_kenning_launch.py
        examples/yolact/launch/yolact_tensorrt_launch.py
        DESTINATION share/${PROJECT_NAME})
endif(BUILD_YOLACT)

#############

install(TARGETS
    basecvnode
    EXPORT basecvnodeTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

export(TARGETS
    basecvnode
    FILE ${PROJECT_BINARY_DIR}/basecvnodeTargets.cmake
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
