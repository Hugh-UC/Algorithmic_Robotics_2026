"""
Pose Graph SLAM Launch

Launches the full SLAM system:
  1. Static transforms (map->odom, base_link->lidar)
  2. SLAM Node — builds pose graph, optimises, publishes corrected map

The SLAM node handles odometry internally, so the motion_model_node
is NOT needed here. (You can run dead_reckoning.launch.py alongside for comparison.)

Usage:
    ros2 launch succulence_rover_ros slam.launch.py

    # For physical robot:
    ros2 launch succulence_rover_ros slam.launch.py \\
        odom_frame:=odom base_link_frame:=base_link lidar_frame:=base_scan

In RViz2 (Fixed Frame: "map"):
  1. Add Map display      -> /succulence/map        (SLAM-corrected, clean!)
  2. Add Path display     -> /succulence/slam/path  (optimised trajectory)
  3. Add Odometry display -> /succulence/slam/odometry

Compare with Week 5 (run both simultaneously):
  - /succulence/map/odom_only = ghost walls (dead reckoning)
  - /succulence/map           = clean walls (SLAM corrected)
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

        # Static transform: map -> odom (identity)
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

        # Static transform: base_link -> lidar (identity — lidar aligned with base)
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

        # SLAM Node
        Node(
            package='succulence_rover_ros',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
