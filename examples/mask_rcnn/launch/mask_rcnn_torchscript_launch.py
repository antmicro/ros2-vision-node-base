# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Launches MaskRCNN node implemented in TorchScript."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from cvnode_base.utils.launch_helper import (
    common_parameters,
    cvnode_manager_gui_node,
    cvnode_manager_node,
    kenning_test_report_node,
)


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description for MaskRCNN node implemented in TorchScript.

    Returns
    -------
    LaunchDescription
        Launch description with all nodes and parameters.
    """
    args, optional_args = common_parameters()

    # Obligatory arguments
    args.append(
        DeclareLaunchArgument(
            "class_names_path",
            description="Path to the file containing classes",
        )
    )

    # Arguments with default values
    scenario, scenario_arg = optional_args["scenario"]
    inference_timeout_ms, timeout_arg = optional_args["inference_timeout_ms"]
    preserve_output, preserve_arg = optional_args["preserve_output"]
    publish_visualizations, gui_arg = optional_args["publish_visualizations"]
    log_level, log_level_arg = optional_args["log_level"]

    args.extend(
        [scenario_arg, timeout_arg, preserve_arg, gui_arg, log_level_arg]
    )

    mask_rcnn_node_container = ComposableNodeContainer(
        name="mask_rcnn_node_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cvnode_base",
                plugin="cvnode_base::MaskRCNNTorchScript",
                name="mask_rcnn_node",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "class_names_path": LaunchConfiguration(
                            "class_names_path"
                        ),
                    }
                ],
            ),
        ],
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            log_level,
        ],
    )

    kenning_node = kenning_test_report_node(log_level)
    gui_node = cvnode_manager_gui_node(log_level)
    manager_node = cvnode_manager_node(
        log_level=log_level,
        visualizations=publish_visualizations,
        timeout=inference_timeout_ms,
        preserve=preserve_output,
        scenario=scenario,
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{time}] [{severity}] [{name}] {function_name}: {message}",  # noqa: E501
            ),
            gui_node,
            kenning_node,
            manager_node,
            mask_rcnn_node_container,
            *args,
        ],
    )
