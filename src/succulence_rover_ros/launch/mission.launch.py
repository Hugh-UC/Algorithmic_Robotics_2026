"""
Full Mission Stack Launch

Launches SLAM + A* planner + pure-pursuit navigator together so the rover
maps the world AND drives itself to the hardcoded goal (Kevin's location
in params_sim.yaml/params_physical.yaml):

    /scan, /odom
         |
         v
    [ slam_node ] --> /succulence/map, /succulence/slam/odometry
         |
         v
    [ planner_node (A*) ] --> /succulence/plan
         |
         v
    [ navigator_node (pure pursuit) ] --> /cmd_vel

Usage:
    ros2 launch succulence_rover_ros mission.launch.py                  # default to simulation
    ros2 launch succulence_rover_ros mission.launch.py mode:=sim
    ros2 launch succulence_rover_ros mission.launch.py mode:=physical


In RViz2 (Fixed Frame: "map"), useful displays:
  - Map        -> /succulence/map
  - Path       -> /succulence/slam/path     (rover's optimised trajectory)
  - Path       -> /succulence/plan          (A* plan to Kevin)
  - Odometry   -> /succulence/slam/odometry
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, GroupAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, EqualsSubstitution, PythonExpression


def generate_launch_description():
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

    # define mode launch argument ('sim' or 'physical')
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description='Target environment: "sim" or "physical"'
    )
    # define costmap mode launch argument ('both', 'global', 'local', or 'none')
    costmap_mode_arg = DeclareLaunchArgument(
        'costmap',
        default_value='both',
        description='Which costmaps to use: "both", "global", "local", or "none"'
    )

    # validate mode argument
    mode = LaunchConfiguration('mode')
    is_physical = EqualsSubstitution(mode, 'physical')
    is_sim = EqualsSubstitution(mode, 'sim')

    costmap_mode = LaunchConfiguration('costmap')

    # load appropriate params file (mode-specified)
    params_file = [config_dir, '/params_', mode, '.yaml']

    # assign default frame names (mode-specific)
    odom_frame_default = PythonExpression(["'odom' if '", mode, "' == 'physical' else 'succulence/odom'"])
    base_link_frame_default = PythonExpression(["'base_link' if '", mode, "' == 'physical' else 'succulence/base_link'"])
    lidar_frame_default = PythonExpression(["'base_scan' if '", mode, "' == 'physical' else 'succulence/lidar_link'"])
    map_frame_default = 'map'       # same for both modes

    # specific launch arguments for frames
    odom_frame_arg = DeclareLaunchArgument('odom_frame', default_value=odom_frame_default, description='Odometry frame')
    base_link_frame_arg = DeclareLaunchArgument('base_link_frame', default_value=base_link_frame_default, description='Base link frame')
    lidar_frame_arg = DeclareLaunchArgument('lidar_frame', default_value=lidar_frame_default, description='Lidar frame')
    map_frame_arg = DeclareLaunchArgument('map_frame', default_value=map_frame_default, description='Map frame')

    # final frame values
    odom_frame = LaunchConfiguration('odom_frame')
    base_link_frame = LaunchConfiguration('base_link_frame')
    lidar_frame = LaunchConfiguration('lidar_frame')
    map_frame = LaunchConfiguration('map_frame')

    # reset pose service call (physical only)
    reset_pose = ExecuteProcess(
        condition=IfCondition(is_physical),
        cmd=['ros2', 'service', 'call', '/reset_pose', 'irobot_create_msgs/srv/ResetPose', '{}'],
        output='screen',
    )

    # Core Nodes
    stack_nodes = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', map_frame, odom_frame],
            output='screen',
        ),

        # Lidar static TF is ONLY for Simulation
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_lidar_publisher',
            arguments=['0', '0', '0', '0', '0', '0', '1', base_link_frame, lidar_frame],
            output='screen',
            condition=IfCondition(is_sim)
        ),

        Node(
            package='succulence_rover_ros',
            executable='slam_node',
            name='slam_node',
            output='screen',
            parameters=[params_file],
        ),

        Node(
            package='succulence_rover_ros',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[
                params_file,
                {'costmap_mode': costmap_mode}
            ],
        ),

        Node(
            package='succulence_rover_ros',
            executable='navigator_node',
            name='navigator_node',
            output='screen',
            parameters=[params_file],
        ),
    ]

    # sim launch (immediate)
    sim_launch = GroupAction(
        condition=IfCondition(is_sim),
        actions=stack_nodes
    )

    # physical launch (waits for reset_pose)
    physical_launch = RegisterEventHandler(
        condition=IfCondition(is_physical),
        event_handler=OnProcessExit(
            target_action=reset_pose,
            on_exit=stack_nodes
        )
    )

    return LaunchDescription([
        mode_arg,
        costmap_mode_arg,
        odom_frame_arg,
        base_link_frame_arg,
        lidar_frame_arg,
        map_frame_arg,
        reset_pose,
        sim_launch,
        physical_launch
    ])
