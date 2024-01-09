# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Launches MaskRCNN node implemented with Detectron2."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    """
    Generates launch description for MaskRCNN node implemented with Detectron2.

    Returns
    -------
    LaunchDescription
        Launch description with all nodes and parameters.
    """
    args = []

    # Obligatory arguments
    args.append(
        DeclareLaunchArgument(
            "class_names_path",
            description="Path to the file containing classes",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "measurements",
            description="Path where measurements should be saved",
        )
    )
    args.append(
        DeclareLaunchArgument(
            "report_path",
            description="Path where rendered inference report should be saved",
        )
    )

    # Optional arguments
    model_path = LaunchConfiguration("model_path", default="COCO")
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

    scenario = LaunchConfiguration("scenario", default="synthetic")
    args.append(
        DeclareLaunchArgument(
            "scenario",
            default_value="synthetic",
            description="Testing scenario strategy. Possible values:\n"
            "synthetic - Manager is waiting for previous frame inference "
            "to finish before sending next frame\n"
            "real_world_last - Manager is sending frames with `inference_timeout_ms` latency, "  # noqa: E501
            "but skip previous frame inference if next frame is ready\n"
            "real_world_first - Manager is sending frames with `inference_timeout_ms` latency, "  # noqa: E501
            "but skip frame if previous frame inference is not finished\n",
        )
    )

    inference_timeout_ms = LaunchConfiguration(
        "inference_timeout_ms", default="300"
    )
    args.append(
        DeclareLaunchArgument(
            "inference_timeout_ms",
            default_value="300",
            description="Inference timeout in milliseconds used by real_world_* scenarios",  # noqa: E501
        )
    )

    preserve_output = LaunchConfiguration("preserve_output", default="False")
    args.append(
        DeclareLaunchArgument(
            "preserve_output",
            default_value="False",
            description="Indicates whether manager should preserve output current frame output."  # noqa: E501
            "Useful for real_world_* scenarios when frame inference is "
            "timeouted so previous frame output is used",
        )
    )

    publish_visualizations = LaunchConfiguration(
        "publish_visualizations", default="True"
    )
    args.append(
        DeclareLaunchArgument(
            "publish_visualizations",
            default_value="True",
            description="Publish visualizations",
        )
    )

    log_level = LaunchConfiguration("log_level", default="INFO")
    args.append(
        DeclareLaunchArgument(
            "log_level",
            default_value="INFO",
            description="Log level",
        )
    )

    config_json = LaunchConfiguration(
        "inference_configuration",
        default="./src/cvnode_base/examples/config/coco_inference.json",
    )
    args.append(
        DeclareLaunchArgument(
            "inference_configuration",
            default_value="./src/cvnode_base/examples/config/coco_inference.json",  # noqa: E501
            description="Path to Kenning's JSON configuration file "
            "with dataset, runtime and protocol specified.",
        )
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

    cvnode_manager_gui_node = ComposableNodeContainer(
        name="cvnode_manager_gui_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cvnode_manager",
                plugin="cvnode_manager::gui::CVNodeManagerGUI",
                name="cvnode_manager_gui",
            ),
        ],
        output="both",
    )

    kenning_node = ExecuteProcess(
        name="kenning_node",
        cmd=[
            [
                "python -m kenning test report ",
                "--json-cfg ",
                config_json,
                " --measurements ",
                LaunchConfiguration("measurements"),
                " --verbosity ",
                log_level,
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
            cvnode_manager_gui_node,
            cvnode_manager_node,
            kenning_node,
            mask_rcnn_node,
            *args,
        ]
    )
