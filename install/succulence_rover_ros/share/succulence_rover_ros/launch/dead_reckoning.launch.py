"""
Dead Reckoning + Occupancy Grid Mapping Launch

Launches:
  1. Static transforms (map->odom, base_link->lidar)
  2. Motion Model Node — dead-reckoning with covariance propagation
  3. Occupancy Grid Mapper — builds map from odometry + laser scans

Usage:
    ros2 launch succulence_rover_ros dead_reckoning.launch.py

    # For physical robot (TurtleBot), override frame names:
    ros2 launch succulence_rover_ros dead_reckoning.launch.py \\
        odom_frame:=odom base_link_frame:=base_link lidar_frame:=base_scan

In RViz2:
  1. Fixed Frame: "map"
  2. Add Map display       -> topic: /succulence/map/odom_only
  3. Add Path display      -> topic: /succulence/dead_reckoning/path
  4. Add Odometry display  -> topic: /succulence/dead_reckoning/odometry
     (enable "Covariance" checkbox to see uncertainty ellipses)
  5. Drive the robot and watch the ghost walls appear!
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os


def generate_launch_description():
    # Config file path
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    params_file = os.path.join(config_dir, 'params.yaml')

    # Launch arguments for frame names (override for different robots)
    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='succulence/odom',
        description='Odometry frame (change to "odom" for TurtleBot)')

    base_link_frame_arg = DeclareLaunchArgument(
        'base_link_frame',
        default_value='succulence/base_link',
        description='Base link frame (change to "base_link" for TurtleBot)')

    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='succulence/lidar_link',
        description='Lidar frame (change to "base_scan" for TurtleBot)')

    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Map frame')

    return LaunchDescription([
        odom_frame_arg,
        base_link_frame_arg,
        lidar_frame_arg,
        map_frame_arg,

        # Static transform: map -> odom (identity — no SLAM correction yet)
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

        # Static transform: base_link -> lidar_link (identity — lidar aligned with base)
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

        # Motion Model Node (dead reckoning + covariance)
        Node(
            package='succulence_rover_ros',
            executable='motion_model_node',
            name='motion_model',
            output='screen',
            parameters=[params_file],
        ),

        # Occupancy Grid Mapper (builds map from odometry + scans)
        Node(
            package='succulence_rover_ros',
            executable='occupancy_grid_mapper_node',
            name='occupancy_grid_mapper',
            output='screen',
            parameters=[params_file],
        ),
    ])
