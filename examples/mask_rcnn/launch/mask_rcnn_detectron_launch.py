# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Launches MaskRCNN node implemented with Detectron2."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from cvnode_base.utils.launch_helper import (
    common_parameters,
    cvnode_manager_gui_node,
    cvnode_manager_node,
    kenning_test_report_node,
)


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description for MaskRCNN node implemented with Detectron2.

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

    # Optional arguments
    model_path = LaunchConfiguration("model_path", default="COCO")
    args = [arg for arg in args if arg.name != "model_path"]
    args.append(
        DeclareLaunchArgument(
            "model_path",
            default_value="COCO",
            description="Path to the file containing weights for Mask R-CNN "
            "model. If parameter is set to 'COCO', weights will be loaded "
            "from the model zoo",
        )
    )

    num_classes = LaunchConfiguration("num_classes", default="80")
    args.append(
        DeclareLaunchArgument(
            "num_classes",
            default_value="80",
            description="Amount of classes in ROI heads of model",
        )
    )

    scenario, scenario_arg = optional_args["scenario"]
    inference_timeout_ms, timeout_arg = optional_args["inference_timeout_ms"]
    preserve_output, preserve_arg = optional_args["preserve_output"]
    publish_visualizations, gui_arg = optional_args["publish_visualizations"]
    log_level, log_level_arg = optional_args["log_level"]

    args.extend(
        [
            scenario_arg,
            timeout_arg,
            preserve_arg,
            gui_arg,
            log_level_arg,
        ]
    )

    mask_rcnn_node = Node(
        package="cvnode_base",
        executable="mask_rcnn_detectron.py",
        name="mask_rcnn_node",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[
            {
                "model_path": model_path,
                "num_classes": num_classes,
                "class_names_path": LaunchConfiguration("class_names_path"),
            }
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
            manager_node,
            gui_node,
            kenning_node,
            mask_rcnn_node,
            *args,
        ]
    )
