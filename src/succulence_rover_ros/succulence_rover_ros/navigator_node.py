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
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation

from .path_follower import PurePursuit


class NavigatorNode(Node):
    """Pure-pursuit navigator wired to the SLAM solutions stack."""

    def __init__(self):
        super().__init__('navigator_node')

        # all required parameter keys (with default fallbacks)
        param_defaults: dict[str, str | int | float | bool] = {
            'plan_topic': '/succulence/plan',
            'odom_topic': '/succulence/slam/odometry',
            'cmd_vel_topic': '/cmd_vel',
            'control.rate_hz': 20.0,
            'control.lookahead': 0.3,
            'control.max_linear_v': 0.2,
            'control.max_angular_v': 1.0,
            'control.goal_tolerance': 0.1,
            'control.plan_timeout': 2.0
        }

        # declare all parameters, with default
        for name, default_val in param_defaults.items():
            self.declare_parameter(name, default_val)

        # tiny helper to fetch values safely and silence 'pylance'
        def get_p(name: str):
            val = self.get_parameter(name).value
            return val if val is not None else param_defaults[name]
        

        plan_topic          = str(get_p('plan_topic'))
        odom_topic          = str(get_p('odom_topic'))
        cmd_vel_topic       = str(get_p('cmd_vel_topic'))

        rate_hz             = float(get_p('control.rate_hz'))
        self.plan_timeout   = float(get_p('control.plan_timeout'))

        self.follower = PurePursuit(
            lookahead=float(get_p('control.lookahead')),
            max_linear_v=float(get_p('control.max_linear_v')),
            max_angular_v=float(get_p('control.max_angular_v')),
            goal_tolerance=float(get_p('control.goal_tolerance')),
        )

        self.pose: np.ndarray | None = None
        self.path: list[tuple[float, float]] = []
        self.last_plan_time: float = 0.0
        self.arrived = False

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(Path, plan_topic, self._plan_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(1.0 / rate_hz, self._tick)

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

    def _tick(self):
        """
        _summary_
        """
        if self.pose is None:
            self._publish_stop()
            return

        now = self.get_clock().now().nanoseconds / 1e9
        stale = (now - self.last_plan_time) > self.plan_timeout

        if self.arrived or stale or not self.path:
            self._publish_stop()
            return

        v, w, arrived = self.follower.compute_cmd(self.pose, self.path)
        if arrived and not self.arrived:
            self.arrived = True
            self._publish_stop()

            # extract final goal coordinate from A* path
            if self.path and len(self.path) > 0:
                goal_x, goal_y = self.path[-1]
            else:
                goal_x, goal_y = self.pose[0], self.pose[1] # fallback

            self.get_logger().info(f'Goal Completed | Success | Arrived at ({goal_x:.2f}, {goal_y:.2f})')
            self.get_logger().info('✅ TurtleBot Goal Complete! Successfully arrived at goal!')
            sys.exit(0)
            return

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

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
