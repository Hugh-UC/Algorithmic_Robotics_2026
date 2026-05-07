"""
Dead Reckoning + Occupancy Grid Mapping (SIM).

Wired for the Unity Mars sim: succulence/* TF frames and /succulence/* topics.
Uses params_sim.yaml.

Usage:
    ros2 launch succulence_rover_ros dead_reckoning_sim.launch.py

In RViz2 (Fixed Frame: "map"):
  - Map      -> /succulence/map/odom_only
  - Path     -> /succulence/dead_reckoning/path
  - Odometry -> /succulence/dead_reckoning/odometry
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    params_file = os.path.join(config_dir, 'params_sim.yaml')

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame', default_value='succulence/odom',
        description='Odometry frame')
    base_link_frame_arg = DeclareLaunchArgument(
        'base_link_frame', default_value='succulence/base_link',
        description='Base link frame')
    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame', default_value='succulence/lidar_link',
        description='Lidar frame')
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
            executable='motion_model_node',
            name='motion_model',
            output='screen',
            parameters=[params_file],
        ),

        Node(
            package='succulence_rover_ros',
            executable='occupancy_grid_mapper_node',
            name='occupancy_grid_mapper',
            output='screen',
            parameters=[params_file],
        ),
    ])
