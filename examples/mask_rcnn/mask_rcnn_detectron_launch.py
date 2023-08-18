# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launches MaskRCNN node implemented with Detectron2."""

    log_level = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Log level'
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
                name='sample_cvnode_manager')
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
        mask_rcnn_node,
        cvnode_manager_node,
        kenning_node,
    ])
