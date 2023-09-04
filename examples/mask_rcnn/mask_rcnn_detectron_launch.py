# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launches MaskRCNN node implemented with Detectron2."""

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
    preserve_output = DeclareLaunchArgument(
            'preserver_output',
            default_value='True',
            description='Indicates whether manager should save output'
    )

    mask_rcnn_node = Node(
            package='cvnode_base',
            executable='mask_rcnn_detectron.py',
            name='mask_rcnn_node',
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
                    'preserve_output': LaunchConfiguration('preserve_output'),
                }],
            )
        ],
        output='both',
        arguments=['--ros-args',
                   '--log-level', LaunchConfiguration('log_level')],
        on_exit=Shutdown()
    )

    cvnode_manager_gui_node = ComposableNodeContainer(
            name='cvnode_manager_gui_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='cvnode_manager',
                    plugin='cvnode_manager::gui::CVNodeManagerGUI',
                    name='cvnode_manager_gui',
                ),
            ],
            output='both',
            on_exit=Shutdown(),
    )

    kenning_node = ExecuteProcess(
        name='kenning_node',
        cmd='python -m kenning test \
--json-cfg ./src/cvnode_base/examples/mask_rcnn/mask_rcnn_ros2_inference.json \
--measurements ./build/ros2-client-measurements.json \
--verbosity INFO'.split(' ')
    )

    return LaunchDescription([
        cvnode_manager_gui_node,
        cvnode_manager_node,
        inference_timeout_ms,
        kenning_node,
        log_level,
        mask_rcnn_node,
        preserve_output,
        publish_visualizations,
        scenario,
    ])
