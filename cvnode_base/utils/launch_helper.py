# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for launch scripts construction."""

from typing import Dict, List, Tuple

from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import LaunchConfigurationNotEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def common_parameters() -> (
    Tuple[
        List[DeclareLaunchArgument],
        Dict[str, Tuple[LaunchConfiguration, DeclareLaunchArgument]],
    ]
):
    """
    Defines common parameters for CVNode launch scripts.

    List of obligatory arguments:
    - model_path - Path to the file to load model from
    - measurements - Path where measurements will be stored
    - report_path - Path to rendered inference report to store
    - inference_configuration - Path to Kenning's JSON configuration file

    List of optional arguments:
    - scenario - Testing scenario strategy
    - inference_timeout_ms - Inference timeout in milliseconds
    - preserve_output - Whether manager should preserve current frame output
    - publish_visualizations - Whether to publish data to visualize
    - device - Device to inference model on
    - log_level - Logging level

    Returns
    -------
    List[DeclareLaunchArgument]
        List of obligatory arguments.
    Dict[str, Tuple[LaunchConfiguration, DeclareLaunchArgument]]
        Dictionary of optional arguments.
    """
    # Obligatory arguments
    obligatory = []
    obligatory.append(
        DeclareLaunchArgument(
            "model_path",
            description="Path to the file to load model from",
        )
    )
    obligatory.append(
        DeclareLaunchArgument(
            "measurements",
            description="Path where measurements will be stored",
        )
    )
    obligatory.append(
        DeclareLaunchArgument(
            "report_path",
            description="Path to rendered inference report to store",
        )
    )

    # Optional arguments
    optional = {}

    scenario = LaunchConfiguration("scenario", default="synthetic")
    optional["scenario"] = (
        scenario,
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
            choices=["synthetic", "real_world_last", "real_world_first"],
        ),
    )

    optional["inference_configuration"] = (
        LaunchConfiguration(
            "inference_configuration",
            default="./src/vision_node_base/examples/config/lindenthal_camera_traps_demo_inference.json",  # noqa: E501
        ),
        DeclareLaunchArgument(
            "inference_configuration",
            description="Path to Kenning's JSON configuration file "
            "with dataset, runtime and protocol specified",
            default_value="./src/vision_node_base/examples/config/lindenthal_camera_traps_demo_inference.json",  # noqa: E501
        ),
    )

    optional["inference_timeout_ms"] = (
        LaunchConfiguration("inference_timeout_ms", default="50"),
        DeclareLaunchArgument(
            "inference_timeout_ms",
            description="Inference timeout in milliseconds used by real_world_* scenarios",  # noqa: E501
            default_value="50",
        ),
    )

    optional["preserve_output"] = (
        LaunchConfiguration("preserve_output", default="True"),
        DeclareLaunchArgument(
            "preserve_output",
            default_value="True",
            description="Indicates whether manager should preserve current "
            "frame output. Useful for real_world_* scenarios when frame "
            "inference is timeouted, so previous frame output is used",
            choices=["True", "False"],
        ),
    )

    optional["publish_visualizations"] = (
        LaunchConfiguration("publish_visualizations", default="True"),
        DeclareLaunchArgument(
            "publish_visualizations",
            description="Indicates whether to publish data for visualization "
            "to ROS2 topic",
            default_value="True",
            choices=["True", "False"],
        ),
    )

    optional["device"] = (
        LaunchConfiguration("device", default="cpu"),
        DeclareLaunchArgument(
            "device",
            description="Device to inference model on",
            default_value="cpu",
            choices=["cuda", "cpu"],
        ),
    )

    optional["log_level"] = (
        LaunchConfiguration("log_level", default="INFO"),
        DeclareLaunchArgument(
            "log_level",
            description="Logging level",
            default_value="INFO",
            choices=["ERROR", "INFO", "DEBUG"],
        ),
    )
    return obligatory, optional


def cvnode_manager_node(
    log_level: LaunchConfiguration,
    visualizations: LaunchConfiguration,
    timeout: LaunchConfiguration,
    scenario: LaunchConfiguration,
    preserve: LaunchConfiguration,
    container_name: str = "cvnode_manager_container",
    node_name: str = "cvnode_manager",
) -> ComposableNodeContainer:
    """
    Composes node container for CVNodeManager.

    Parameters
    ----------
    log_level : LaunchConfiguration
        Logging level.
    visualizations : LaunchConfiguration
        Whether to publish data for visualization to ROS2 topic.
    timeout : LaunchConfiguration
        Inference timeout in milliseconds.
    scenario : LaunchConfiguration
        Testing scenario strategy.
    preserve : LaunchConfiguration
        Whether manager should preserve current frame output.
    container_name : str, optional
        Name of the container, by default "cvnode_manager_container".
    node_name : str, optional
        Name of the node, by default "cvnode_manager".

    Returns
    -------
    ComposableNodeContainer
        Composed node container.
    """
    return ComposableNodeContainer(
        name=container_name,
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cvnode_manager",
                plugin="cvnode_manager::CVNodeManager",
                name=node_name,
                parameters=[
                    {
                        "publish_visualizations": visualizations,
                        "inference_timeout_ms": timeout,
                        "scenario": scenario,
                        "preserve_output": preserve,
                    }
                ],
            )
        ],
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            log_level,
        ],
    )


def kenning_test_report_node(
    log_level: LaunchConfiguration,
    inference_configuration: LaunchConfiguration,
    node_name: str = "kenning_node",
) -> ExecuteProcess:
    """
    Creates a launch action for kenning test report node.

    Parameters
    ----------
    log_level : LaunchConfiguration
        Logging level.
    inference_configuration: LaunchConfiguration
        Used inference test scenario for Kenning
    node_name : str
        Name of the node.

    Returns
    -------
    ExecuteProcess
        Launch action for kenning test report node.
    """
    return ExecuteProcess(
        name=node_name,
        cmd=[
            [
                "python -m kenning test report ",
                "--json-cfg ",
                inference_configuration,
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


def cvnode_manager_gui_node(
    log_level: LaunchConfiguration,
    container_name: str = "cvnode_manager_gui_container",
    node_name: str = "cvnode_manager_gui",
) -> ComposableNodeContainer:
    """
    Composes node container for CVNodeManagerGUI.

    Parameters
    ----------
    log_level : LaunchConfiguration
        Logging level.
    container_name : str
        Name of the container.
    node_name : str
        Name of the node.

    Returns
    -------
    ComposableNodeContainer
        Composed node container.
    """
    return ComposableNodeContainer(
        name=container_name,
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="cvnode_manager",
                plugin="cvnode_manager::gui::CVNodeManagerGUI",
                name=node_name,
            ),
        ],
        output="both",
        arguments=[
            "--ros-args",
            "--log-level",
            log_level,
        ],
        condition=LaunchConfigurationNotEquals(
            "publish_visualizations", "False"
        ),
    )
