"""
Navigator ROS2 Node

Subscribes to the planner's path and the SLAM odometry. On a fixed timer,
runs the pure-pursuit controller and publishes /cmd_vel. Publishes zero
velocity when there is no plan, when the plan is stale, or when the robot
has arrived at the goal.
"""

import sys
import numpy as np
import rclpy
from typing import Any
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

from .path_follower import PurePursuit


class NavigatorNode(Node):
    """Pure-pursuit navigator wired to the SLAM solutions stack."""

    def __init__(self):
        """
        _summary_
        """
        super().__init__('navigator_node')

        # all required parameter keys (with default fallbacks)
        param_defaults: dict[str, str | int | float | bool] = {
            'plan_topic': '/succulence/plan',
            'odom_topic': '/succulence/slam/odometry',
            'scan_topic': '/scan',                      # Physical raw Lidar
            'cmd_vel_topic': '/cmd_vel_unstamped',      # Physical TB4 unstamped velocity
            
            # Emergency Braking
            'safety.mode': 'both',                      # Options: 'both', 'collision', 'emergency', 'none'
            'safety.collision_brake_dist': 0.25,        # Threshold for soft recovery (m)
            'safety.emergency_brake_dist': 0.12,        # Threshold for hard motor lock (m)
            'safety.forward_cone_angle': 0.7,           # Width of protective Lidar cone (rad)
            'safety.recovery_spin_v': 0.3,              # Cap on turn speed during recovery (rad/s)

            'lidar.yaw_offset': 1.570,                  # Physical TB4 Lidar rotation
            
            # Adaptive Pure Pursuit
            'control.rate_hz': 15.0,                    # Physical network limit
            'control.lookahead_min': 0.2,
            'control.lookahead_max': 0.6,
            'control.lookahead_ratio': 1.5,
            'control.max_linear_v': 0.3,
            'control.max_angular_v': 0.18,
            'control.max_linear_a': 0.5,                # Max forward/reverse acceleration (m/s^2)
            'control.max_angular_a': 1.0,               # Max turning acceleration (rad/s^2)
            'control.goal_tolerance': 0.05,             # Account for physical overshoot
            'control.plan_timeout': 6.0
        }

        # declare all parameters, with default
        for name, default_val in param_defaults.items():
            self.declare_parameter(name, default_val)

        # tiny helper to fetch values safely and silence 'pylance'
        def get_p(name: str) -> Any:
            """
            _summary_

            Args:
                name (str): _description_

            Returns:
                Any: _description_
            """
            val = self.get_parameter(name).value
            return val if val is not None else param_defaults[name]
        
        # topic names
        plan_topic          = str(get_p('plan_topic'))
        odom_topic          = str(get_p('odom_topic'))
        scan_topic          = str(get_p('scan_topic'))
        cmd_vel_topic       = str(get_p('cmd_vel_topic'))

        # emergency braking
        self.safety_mode        = str(get_p('safety.mode')).lower()
        self.col_brake_dist     = float(get_p('safety.collision_brake_dist'))
        self.emg_brake_dist     = float(get_p('safety.emergency_brake_dist'))
        self.cone_angle_rad     = float(get_p('safety.forward_cone_angle'))
        self.recovery_spin_v    = float(get_p('safety.recovery_spin_v'))
        self.emergency_stop     = False         # track emergency stop state
        self.collision_stop     = False         # track collision stop state

        self.lidar_yaw_offset = float(get_p('lidar.yaw_offset'))

        # rate/timeout
        self.rate_hz        = float(get_p('control.rate_hz'))
        self.plan_timeout   = float(get_p('control.plan_timeout'))

        # velocity/acceleration & goal tolerance
        max_linear_v : float            = float(get_p('control.max_linear_v'))
        max_angular_v : float           = float(get_p('control.max_angular_v'))
        self.max_linear_a : float       = float(get_p('control.max_linear_a'))
        self.max_angular_a : float      = float(get_p('control.max_angular_a'))
        goal_tolerance                  = float(get_p('control.goal_tolerance'))

        # store the adaptive clamps
        self.lookahead_min = float(get_p('control.lookahead_min'))
        self.lookahead_max = float(get_p('control.lookahead_max'))
        self.lookahead_ratio = float(get_p('control.lookahead_ratio'))

        # initialize follower
        self.follower = PurePursuit(
            lookahead=self.lookahead_max,       # default starting value
            max_linear_v=max_linear_v,
            max_angular_v=max_angular_v,
            max_linear_a=self.max_linear_a,
            max_angular_a=self.max_angular_a,
            recovery_spin_v=self.recovery_spin_v,
            goal_tolerance=goal_tolerance,
        )

        self.pose : np.ndarray | None           = None
        self.path : list[tuple[float, float]]   = []
        self.last_plan_time : float             = 0.0
        self.arrived                            = False
        
        # publishers/subscribers
        self.cmd_pub    = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(Marker, '/succulence/lookahead_visual', 10)
        self.create_subscription(Path, plan_topic, self._plan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f'NavigatorNode started — plan: {plan_topic}, odom: {odom_topic}, '
            f'cmd_vel: {cmd_vel_topic}')

    def _plan_cb(self, msg: Path):
        """
        _summary_

        Args:
            msg (Path): _description_
        """
        self.path = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        self.last_plan_time = self.get_clock().now().nanoseconds / 1e9
        if self.path:
            self.arrived = False  # Fresh plan — resume driving.

    def _odom_cb(self, msg: Odometry):
        """
        _summary_

        Args:
            msg (Odometry): _description_
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        self.pose = np.array([x, y, theta])
    
    def _scan_cb(self, msg: LaserScan):
        """
        Monitors a forward cone for immediate collision threats.

        Args:
            msg (LaserScan): _description_
        """
        ranges = np.array(msg.ranges)
        
        # filter out infinite or invalid readings
        valid_mask = np.isfinite(ranges) & (ranges > msg.range_min)
        
        # calculate angle of every beam in scan
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        # apply Lidar mounting offset so 0 radians is 'Robot Forward'
        robot_relative_angles = angles + self.lidar_yaw_offset
        
        # wrap angles cleanly to [-pi, pi]
        robot_relative_angles = (robot_relative_angles + np.pi) % (2 * np.pi) - np.pi
        
        # create mask for only beams inside protective forward cone
        half_cone = self.cone_angle_rad / 2.0
        in_cone_mask = np.abs(robot_relative_angles) <= half_cone
        
        # extract distances of objects directly in front of robot
        cone_ranges = ranges[valid_mask & in_cone_mask]
        
        # safety state flags
        new_col_state = False
        new_emg_state = False

        # parse parameter string into booleans
        enable_emg = self.safety_mode in ['both', 'emergency']
        enable_col = self.safety_mode in ['both', 'collision']
        
        if len(cone_ranges) > 0:
            min_dist = np.min(cone_ranges)
            
            # Check Hard Emergency First (Highest Priority)
            if enable_emg and min_dist < self.emg_brake_dist:
                new_emg_state = True
            # Check Soft Collision Second
            elif enable_col and min_dist < self.col_brake_dist:
                new_col_state = True

        # Logging State Changes (prevents spam)
        if new_emg_state and not self.emergency_stop:
            self.get_logger().error('🚨 CRITICAL EMERGENCY BRAKE! Motors Locked.')
        elif new_col_state and not self.collision_stop:
            self.get_logger().warn('⚠️ Collision Brake! Forward motion stopped, calculating detour.')
        elif not new_col_state and not new_emg_state and (self.collision_stop or self.emergency_stop):
            self.get_logger().info('✅ Safety Zone Clear. Resuming navigation.')

        self.emergency_stop = new_emg_state
        self.collision_stop = new_col_state

    def _tick(self):
        """
        Calculates and publishes motor commands on a fixed timer.
        """
        if self.pose is None:
            self._publish_stop()
            return
        
        # extrapolate dynamic lookahead (with clamping)
        dynamic_lookahead = abs(self.follower.current_v) * self.lookahead_ratio
        dynamic_lookahead = np.clip(dynamic_lookahead, self.lookahead_min, self.lookahead_max)
        self.follower.lookahead = float(dynamic_lookahead)

        # publish visualization to RViz2
        self._publish_lookahead_visual(dynamic_lookahead)

        now = self.get_clock().now().nanoseconds / 1e9
        stale = (now - self.last_plan_time) > self.plan_timeout

        if self.arrived or stale or not self.path:
            self.follower.compute_cmd(self.pose, [], 1.0/self.rate_hz, e_stop=True)
            self._publish_stop()
            return

        v, w, arrived = self.follower.compute_cmd(
            pose=self.pose, 
            path=self.path, 
            dt=(1.0 / self.rate_hz), 
            c_stop=self.collision_stop, 
            e_stop=self.emergency_stop
        )

        if arrived and not self.arrived:
            self.arrived = True
            self._publish_stop()

            # extract final goal coordinate from A* path
            if self.path and len(self.path) > 0:
                goal_x, goal_y = self.path[-1]
            else:
                goal_x, goal_y = self.pose[0], self.pose[1] # fallback

            self.get_logger().info(f'Goal Completed | Success | Arrived at ({goal_x:.4f}, {goal_y:.4f})')
            self.get_logger().info('✅ TurtleBot Goal Complete! Successfully arrived at goal!')
            sys.exit(0)
            return

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

    def _publish_lookahead_visual(self, lookahead: float):
        """
        Publishes a cyan cylinder marker to RViz2 to visualize the 
        current dynamic lookahead radius.

        Args:
            lookahead (float): _description_
        """
        if self.pose is None:
            return
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dynamic_lookahead"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.pose[0])
        marker.pose.position.y = float(self.pose[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Cylinder diameter is 2x the radius
        marker.scale.x = float(lookahead * 2.0)
        marker.scale.y = float(lookahead * 2.0)
        marker.scale.z = 0.02
        
        marker.color.r = 0.0    # Cyan
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.4    # 40% opacity
        
        self.marker_pub.publish(marker)

    def _publish_stop(self):
        """
        _summary_
        """
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
