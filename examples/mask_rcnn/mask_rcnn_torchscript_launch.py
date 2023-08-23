# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launches MaskRCNN node implemented in TorchScript."""

    model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the TorchScript model file'
    )
    log_level = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Log level'
    )
    publish_visualizations = DeclareLaunchArgument(
        'publish_visualizations',
        default_value='False',
        description='Publish visualizations'
    )
    scenario = DeclareLaunchArgument(
        'scenario',
        default_value='synthetic',
        description='Testing scenario strategy'
    )
    inference_timeout_ms = DeclareLaunchArgument(
        'inference_timeout_ms',
        default_value="300",
        description='Inference timeout in milliseconds'
    )

    mask_rcnn_node_container = ComposableNodeContainer(
        name='mask_rcnn_node_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='cvnode_base',
                plugin='cvnode_base::MaskRCNNTorchScript',
                name='mask_rcnn_node',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path')
                }],
            ),
        ],
        output='both',
        arguments=['--ros-args',
                   '--log-level', LaunchConfiguration('log_level')]
    )

    cvnode_manager_node = ComposableNodeContainer(
        name='cvnode_manager_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='cvnode_manager',
                plugin='cvnode_manager::CVNodeManager',
                name='sample_cvnode_manager',
                parameters=[{
                    'publish_visualizations': LaunchConfiguration('publish_visualizations'),    # noqa: E501
                    'inference_timeout_ms': LaunchConfiguration('inference_timeout_ms'),        # noqa: E501
                    'scenario': LaunchConfiguration('scenario'),
                }],
            )
        ],
        output='both',
        arguments=['--ros-args',
                   '--log-level', LaunchConfiguration('log_level')],
        on_exit=Shutdown()
    )

    kenning_node = ExecuteProcess(
        name='kenning_node',
        cmd='python -m kenning test \
--json-cfg ./src/cvnode_base/examples/mask_rcnn/mask_rcnn_ros2_inference.json \
--measurements ./build/ros2-client-measurements.json \
--verbosity INFO'.split(' ')
    )

    return LaunchDescription([
        log_level,
        model_path,
        publish_visualizations,
        scenario,
        inference_timeout_ms,
        mask_rcnn_node_container,
        cvnode_manager_node,
        kenning_node,
    ])
