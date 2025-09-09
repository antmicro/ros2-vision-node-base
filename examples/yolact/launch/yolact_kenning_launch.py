# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Launches YOLACT CVNode with CVNodeManager and Kenning."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    Shutdown
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from cvnode_base.utils.launch_helper import (
    common_parameters,
    cvnode_manager_gui_node,
    cvnode_manager_node
)


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description with all the nodes.

    Returns
    -------
    LaunchDescription
        Launch description with all the nodes.
    """
    args, optional_args = common_parameters()

    # Obligatory arguments
    args.append(
        DeclareLaunchArgument(
            "backend",
            description="Backend to execute YOLACT model on"
            "Possible values: tvm, onnxruntime, tflite",
            choices=["tvm", "onnxruntime", "tflite"],
        )
    )

    # Optional arguments
    scenario, scenario_arg = optional_args["scenario"]
    inference_configuration, inference_configuration_arg = optional_args[
        "inference_configuration"
    ]
    inference_timeout_ms, timeout_arg = optional_args["inference_timeout_ms"]
    preserve_output, preserve_arg = optional_args["preserve_output"]
    publish_visualizations, gui_arg = optional_args["publish_visualizations"]
    device, device_arg = optional_args["device"]
    log_level, log_level_arg = optional_args["log_level"]

    args.extend(
        [
            scenario_arg,
            inference_configuration_arg,
            timeout_arg,
            preserve_arg,
            gui_arg,
            device_arg,
            log_level_arg,
        ]
    )

    # Nodes
    yolact_node = Node(
        package="cvnode_base",
        executable="yolact_kenning.py",
        name="yolact_node",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
                "device": device,
            }
        ],
    )

    kenning_node = Node(
        executable="kenning",
        name="kenning_node",
        arguments=["ros","test","report","--verbosity",log_level,"--ros-args","--log-level",log_level],
        parameters=[
            {
                "config_file":inference_configuration,
                "measurements":LaunchConfiguration("measurements"),
                "report_types": ["detection"],
                "report_path": LaunchConfiguration("report_path"),
                "to_html":"True"
            }
        ],
        on_exit=Shutdown(),
    )

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
            yolact_node,
            *args,
        ]
    )
