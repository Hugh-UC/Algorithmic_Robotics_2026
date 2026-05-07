"""
Pose Graph SLAM Launch (Week 11)

Usage:
    ros2 launch succulence_rover_ros slam.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    params_file = os.path.join(config_dir, 'params.yaml')

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame', default_value='succulence/odom',
        description='Odometry frame (change to "odom" for TurtleBot)')
    base_link_frame_arg = DeclareLaunchArgument(
        'base_link_frame', default_value='succulence/base_link',
        description='Base link frame (change to "base_link" for TurtleBot)')
    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame', default_value='succulence/lidar_link',
        description='Lidar frame (change to "base_scan" for TurtleBot)')
    map_frame_arg = DeclareLaunchArgument(
        'map_frame', default_value='map', description='Map frame')

    return LaunchDescription([
        odom_frame_arg,
        base_link_frame_arg,
        lidar_frame_arg,
        map_frame_arg,

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=[
                '0', '0', '0',
                '0', '0', '0', '1',
                LaunchConfiguration('map_frame'),
                LaunchConfiguration('odom_frame')
            ],
            output='screen',
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_lidar_publisher',
            arguments=[
                '0', '0', '0',
                '0', '0', '0', '1',
                LaunchConfiguration('base_link_frame'),
                LaunchConfiguration('lidar_frame')
            ],
            output='screen',
        ),

        Node(
            package='succulence_rover_ros',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
