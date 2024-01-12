# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Launches MaskRCNN node implemented in TorchScript."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description for MaskRCNN node implemented in TorchScript.

    Returns
    -------
    LaunchDescription
        Launch description with all nodes and parameters.
    """
    model_path = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to the TorchScript model file",
    )
    log_level = DeclareLaunchArgument(
        "log_level", default_value="INFO", description="Log level"
    )
    scenario = DeclareLaunchArgument(
        "scenario",
        default_value="synthetic",
        description="Testing scenario strategy",
    )
    inference_timeout_ms = DeclareLaunchArgument(
        "inference_timeout_ms",
        default_value="300",
        description="Inference timeout in milliseconds",
    )
    preserve_output = DeclareLaunchArgument(
        "preserve_output",
        default_value="True",
        description="Indicates whether manager should save output",
    )
    class_names_path = DeclareLaunchArgument(
        "class_names_path", description="Path to the file containing classes"
    )
    measurements = DeclareLaunchArgument(
        "measurements", description="Path where measurements should be saved"
    )
    report_path = DeclareLaunchArgument(
        "report_path",
        description="Path where rendered inference report should be saved",
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
                        ),  # noqa: E501
                    }
                ],
            ),
        ],
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            LaunchConfiguration("log_level"),
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
                        "inference_timeout_ms": LaunchConfiguration(
                            "inference_timeout_ms"
                        ),  # noqa: E501
                        "scenario": LaunchConfiguration("scenario"),
                        "preserve_output": LaunchConfiguration(
                            "preserve_output"
                        ),
                    }
                ],
            )
        ],
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            LaunchConfiguration("log_level"),
        ],
    )

    kenning_node = ExecuteProcess(
        name="kenning_node",
        cmd=[
            [
                "python -m kenning test report ",
                "--json-cfg ./src/cvnode_base/examples/mask_rcnn/config/mask_rcnn_ros2_inference.json "  # noqa: E501
                "--measurements ",
                LaunchConfiguration("measurements"),
                " --verbosity ",
                LaunchConfiguration("log_level"),
                " --report-types detection ",
                "--report-path ",
                LaunchConfiguration("report_path"),
                " --to-html",
            ]
        ],
        on_exit=Shutdown(),
        shell=True,
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="RCUTILS_CONSOLE_OUTPUT_FORMAT",
                value="[{time}] [{severity}] [{name}] {function_name}: {message}",  # noqa: E501
            ),
            class_names_path,
            cvnode_manager_node,
            inference_timeout_ms,
            kenning_node,
            log_level,
            mask_rcnn_node_container,
            measurements,
            model_path,
            preserve_output,
            report_path,
            scenario,
        ]
    )
