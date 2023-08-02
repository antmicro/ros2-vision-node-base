# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launches MaskRCNN node implemented in TorchScript."""
    model_path = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the TorchScript model file'
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
            )
        ],
        output='both',
    )
    return LaunchDescription([
        model_path,
        mask_rcnn_node_container,
    ])
