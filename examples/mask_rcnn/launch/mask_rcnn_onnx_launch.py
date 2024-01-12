# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Launches MaskRCNN node with ONNXRuntime backend."""

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from cvnode_base.utils.launch import (
    common_parameters,
    cvnode_manager_gui_node,
    kenning_test_report_node,
)


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description for MaskRCNN node implemented with ONNX.

    Returns
    -------
    LaunchDescription
        Launch description with all nodes and parameters.
    """
    args, optional_args = common_parameters()

    # Arguments with default values
    scenario, scenario_arg = optional_args["scenario"]
    inference_timeout_ms, timeout_arg = optional_args["inference_timeout_ms"]
    preserve_output, preserve_arg = optional_args["preserve_output"]
    publish_visualizations, gui_arg = optional_args["publish_visualizations"]
    device, device_arg = optional_args["device"]
    log_level, log_level_arg = optional_args["log_level"]

    args.extend(
        [
            scenario_arg,
            timeout_arg,
            preserve_arg,
            gui_arg,
            device_arg,
            log_level_arg,
        ]
    )

    kenning_node = kenning_test_report_node(log_level)
    gui_node = cvnode_manager_gui_node()

    mask_rcnn_node = Node(
        package="cvnode_base",
        executable="mask_rcnn_onnx.py",
        name="mask_rcnn_node",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "class_names_path": LaunchConfiguration("class_names_path"),
                "device": device,
            }
        ],
    )

    cvnode_manager_node = ComposableNodeContainer(
        name="cvnode_manager_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cvnode_manager",
                plugin="cvnode_manager::CVNodeManager",
                name="sample_cvnode_manager",
                parameters=[
                    {
                        "publish_visualizations": publish_visualizations,
                        "inference_timeout_ms": inference_timeout_ms,
                        "scenario": scenario,
                        "preserve_output": preserve_output,
                    }
                ],
            )
        ],
        output="both",
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{time}] [{severity}] [{name}] {function_name}: {message}",  # noqa: E501
            ),
            cvnode_manager_node,
            gui_node,
            kenning_node,
            mask_rcnn_node,
            *args,
        ]
    )
