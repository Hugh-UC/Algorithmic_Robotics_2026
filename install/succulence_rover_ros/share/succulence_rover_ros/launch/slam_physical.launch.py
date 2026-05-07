"""
Pose Graph SLAM (PHYSICAL TurtleBot).

Wired for the iRobot Create 3 + RPLidar TurtleBot: /odom, /scan, the
`odom`/`base_link`/`base_scan` TF frames published by the robot driver,
and a static map -> odom publisher. Uses params_physical.yaml.

Startup sequence:
    1. Call /reset_pose so the base's published odometry restarts at zero.
    2. Once the service call returns, bring up the map -> odom static TF
       and the SLAM node — neither subscribes to /odom before the reset
       has taken effect.

Usage:
    ros2 launch succulence_rover_ros slam_physical.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    params_file = os.path.join(config_dir, 'params_physical.yaml')

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame', default_value='odom', description='Odometry frame')
    base_link_frame_arg = DeclareLaunchArgument(
        'base_link_frame', default_value='base_link',
        description='Base link frame')
    map_frame_arg = DeclareLaunchArgument(
        'map_frame', default_value='map', description='Map frame')

    reset_pose = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/reset_pose',
             'irobot_create_msgs/srv/ResetPose', '{}'],
        output='screen',
    )

    stack_nodes = [
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
            package='succulence_rover_ros',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[params_file],
        ),
    ]

    return LaunchDescription([
        odom_frame_arg,
        base_link_frame_arg,
        map_frame_arg,
        reset_pose,
        RegisterEventHandler(
            OnProcessExit(target_action=reset_pose, on_exit=stack_nodes),
        ),
    ])
