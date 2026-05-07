"""
Navigator ROS2 Node

Subscribes to the planner's path and the SLAM odometry. On a fixed timer,
runs the pure-pursuit controller and publishes /cmd_vel. Publishes zero
velocity when there is no plan, when the plan is stale, or when the robot
has arrived at the goal.
"""

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

        self.declare_parameter('plan_topic')
        self.declare_parameter('odom_topic')
        self.declare_parameter('cmd_vel_topic')
        self.declare_parameter('control.rate_hz')
        self.declare_parameter('control.lookahead')
        self.declare_parameter('control.max_linear_v')
        self.declare_parameter('control.max_angular_v')
        self.declare_parameter('control.goal_tolerance')
        self.declare_parameter('control.plan_timeout')

        plan_topic = self.get_parameter('plan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        rate_hz = float(self.get_parameter('control.rate_hz').value)
        self.plan_timeout = float(self.get_parameter('control.plan_timeout').value)

        self.follower = PurePursuit(
            lookahead=float(self.get_parameter('control.lookahead').value),
            max_linear_v=float(self.get_parameter('control.max_linear_v').value),
            max_angular_v=float(self.get_parameter('control.max_angular_v').value),
            goal_tolerance=float(self.get_parameter('control.goal_tolerance').value),
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
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.last_plan_time = self.get_clock().now().nanoseconds / 1e9
        if self.path:
            self.arrived = False  # Fresh plan — resume driving.

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        self.pose = np.array([x, y, theta])

    def _tick(self):
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
            self.get_logger().info('Arrived at goal — stopping.')
            self._publish_stop()
            return

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

    def _publish_stop(self):
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
