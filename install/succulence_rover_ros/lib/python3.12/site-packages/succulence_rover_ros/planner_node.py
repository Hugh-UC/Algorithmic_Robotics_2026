"""
A* Planner ROS2 Node

Subscribes to the SLAM occupancy grid and the SLAM odometry. On a timer,
runs A* from the robot's current cell to the hardcoded goal cell and
publishes a nav_msgs/Path in the map frame.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg, Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation

from .astar import astar_search, inflate_obstacles


class PlannerNode(Node):
    """ROS2 wrapper around A*. Replans at a fixed rate against the latest SLAM map."""

    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('map_topic')
        self.declare_parameter('odom_topic')
        self.declare_parameter('plan_topic')
        self.declare_parameter('frames.map_frame')
        self.declare_parameter('planning.replan_period')
        self.declare_parameter('planning.occupancy_threshold')
        self.declare_parameter('planning.treat_unknown_as_obstacle')
        self.declare_parameter('planning.inflation_radius_cells')
        self.declare_parameter('goal.x')
        self.declare_parameter('goal.y')

        map_topic = self.get_parameter('map_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        plan_topic = self.get_parameter('plan_topic').value

        self.map_frame = self.get_parameter('frames.map_frame').value
        self.replan_period = self.get_parameter('planning.replan_period').value
        self.occ_threshold = int(self.get_parameter('planning.occupancy_threshold').value)
        self.unknown_as_obstacle = bool(
            self.get_parameter('planning.treat_unknown_as_obstacle').value)
        self.inflation_radius = int(
            self.get_parameter('planning.inflation_radius_cells').value)

        self.goal_x = float(self.get_parameter('goal.x').value)
        self.goal_y = float(self.get_parameter('goal.y').value)

        self.latest_map: OccupancyGridMsg | None = None
        self.robot_xy: tuple[float, float] | None = None
        self.consecutive_failures = 0

        self.path_pub = self.create_publisher(Path, plan_topic, 10)
        self.create_subscription(OccupancyGridMsg, map_topic, self._map_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(self.replan_period, self._replan)

        self.get_logger().info(
            f'PlannerNode started — map: {map_topic}, odom: {odom_topic}, '
            f'goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')

    def _map_cb(self, msg: OccupancyGridMsg):
        self.latest_map = msg

    def _odom_cb(self, msg: Odometry):
        self.robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _world_to_cell(self, x: float, y: float, info) -> tuple[int, int]:
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        return row, col

    def _cell_to_world(self, row: int, col: int, info) -> tuple[float, float]:
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def _replan(self):
        if self.latest_map is None or self.robot_xy is None:
            return

        info = self.latest_map.info
        grid = np.frombuffer(bytes(self.latest_map.data), dtype=np.int8).reshape(
            info.height, info.width).copy()

        blocked = inflate_obstacles(
            grid, self.inflation_radius,
            self.occ_threshold, self.unknown_as_obstacle)

        start = self._world_to_cell(*self.robot_xy, info)
        goal = self._world_to_cell(self.goal_x, self.goal_y, info)

        if not (0 <= start[0] < info.height and 0 <= start[1] < info.width):
            self._log_failure('robot outside map bounds')
            self._publish_empty_path()
            return
        if not (0 <= goal[0] < info.height and 0 <= goal[1] < info.width):
            self._log_failure('goal outside map bounds')
            self._publish_empty_path()
            return

        # If the start cell was inflated shut (robot on the edge of a blocked
        # cell), unblock it so A* can at least escape.
        if blocked[start]:
            blocked[start] = False

        path_cells = astar_search(blocked, start, goal)
        if path_cells is None:
            self._log_failure('no path found')
            self._publish_empty_path()
            return

        self.consecutive_failures = 0
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame

        # Identity orientation for all waypoints — the controller handles heading.
        q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        for (r, c) in path_cells:
            wx, wy = self._cell_to_world(r, c, info)
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation = q
            path_msg.poses.append(ps)

        # Replace the final waypoint with the exact goal so the controller
        # converges to the requested position rather than the cell centre.
        if path_msg.poses:
            path_msg.poses[-1].pose.position.x = self.goal_x
            path_msg.poses[-1].pose.position.y = self.goal_y

        self.path_pub.publish(path_msg)

    def _publish_empty_path(self):
        empty = Path()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = self.map_frame
        self.path_pub.publish(empty)

    def _log_failure(self, reason: str):
        self.consecutive_failures += 1
        # Log the first failure and then once every ~10 to avoid spam.
        if self.consecutive_failures == 1 or self.consecutive_failures % 10 == 0:
            self.get_logger().warn(
                f'Planner: {reason} (failure #{self.consecutive_failures})')


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
