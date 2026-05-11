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
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation

from .astar import astar_search, inflate_obstacles


class PlannerNode(Node):
    """ROS2 wrapper around A*. Replans at a fixed rate against the latest SLAM map."""

    def __init__(self):
        """
        Subscribes to SLAM map and odometry, publishes A* path.
        """
        super().__init__('planner_node')

        self.declare_parameter('map_topic')
        self.declare_parameter('odom_topic')
        self.declare_parameter('plan_topic')
        self.declare_parameter('frames.map_frame')
        self.declare_parameter('costmap_topic', '/succulence/costmap')
        self.declare_parameter('scan_topic', '/succulence/scan')

        # Planning
        self.declare_parameter('planning.replan_period')
        self.declare_parameter('planning.heuristic_weight', 1.2)
        self.declare_parameter('planning.data_weight', 0.1)
        self.declare_parameter('planning.smooth_weight', 0.5)

        # Costmaps
        self.declare_parameter('costmaps.occupancy_threshold')
        self.declare_parameter('costmaps.treat_unknown_as_obstacle')

        self.declare_parameter('costmaps.global.inflation_radius_cells')
        self.declare_parameter('costmaps.global.inflation_weight', 50.0)

        self.declare_parameter('costmaps.local.inflation_radius_cells')
        self.declare_parameter('costmaps.local.inflation_weight', 15.0)
        self.declare_parameter('costmaps.local.max_obstacle_range', 3.0)
        self.declare_parameter('costmaps.local.min_obstacle_range', 0.1)

        self.declare_parameter('goal.x')
        self.declare_parameter('goal.y')
        self.declare_parameter('goal.tolerance', 0.05)

        map_topic = self.get_parameter('map_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        plan_topic = self.get_parameter('plan_topic').value
        costmap_topic = self.get_parameter('costmap_topic').value
        scan_topic = self.get_parameter('scan_topic').value

        self.map_frame = self.get_parameter('frames.map_frame').value
        self.replan_period = self.get_parameter('planning.replan_period').value
        
        # Save costmap settings
        self.occ_threshold = int(self.get_parameter('costmaps.occupancy_threshold').value)
        self.unknown_as_obstacle = bool(self.get_parameter('costmaps.treat_unknown_as_obstacle').value)
        
        self.global_inf_radius = float(self.get_parameter('costmaps.global.inflation_radius_cells').value)
        self.global_inf_weight = float(self.get_parameter('costmaps.global.inflation_weight').value)
        
        self.local_inf_radius = float(self.get_parameter('costmaps.local.inflation_radius_cells').value)
        self.local_inf_weight = float(self.get_parameter('costmaps.local.inflation_weight').value)
        self.local_max_range = float(self.get_parameter('costmaps.local.max_obstacle_range').value)
        self.local_min_range = float(self.get_parameter('costmaps.local.min_obstacle_range').value)

        self.goal_x = float(self.get_parameter('goal.x').value)
        self.goal_y = float(self.get_parameter('goal.y').value)
        self.goal_tolerance = float(self.get_parameter('goal.tolerance').value)

        self.global_costmap: np.ndarray | None = None
        self.global_map_info = None
        self.latest_scan: LaserScan | None = None
        
        self.robot_xy: tuple[float, float] | None = None
        self.robot_theta = 0.0
        self.consecutive_failures = 0

        self.path_pub = self.create_publisher(Path, plan_topic, 10)
        self.costmap_pub = self.create_publisher(OccupancyGridMsg, costmap_topic, 10)

        self.create_subscription(OccupancyGridMsg, map_topic, self._map_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        
        self.create_timer(self.replan_period, self._replan)

        self.get_logger().info(
            f'PlannerNode started — map: {map_topic}, odom: {odom_topic}, '
            f'goal: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        
    def _scan_cb(self, msg: LaserScan):
        """
        Callback for handling incoming laser scan messages.

        Args:
            msg (LaserScan): incoming laser scan message.
        """
        self.latest_scan = msg

    def _map_cb(self, msg: OccupancyGridMsg):
        """
        Callback for handling incoming occupancy grid messages.
        Bakes the heavy Global Costmap in the background.

        Args:
            msg (OccupancyGridMsg): incoming occupancy grid message.
        """
        info = msg.info
        grid = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(info.height, info.width).copy()

        self.global_costmap = inflate_obstacles(
            grid, self.global_inf_radius,
            self.occ_threshold, self.unknown_as_obstacle,
            inflation_weight=self.global_inf_weight
        )
        self.global_map_info = info
        self.latest_map = msg

    def _odom_cb(self, msg: Odometry):
        """
        Callback for handling incoming odometry messages.

        Args:
            msg (Odometry): incoming odometry message.
        """
        self.robot_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # extract yaw (theta) from quaternion, to project laser scan
        q = msg.pose.pose.orientation
        self.robot_theta = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

    def _world_to_cell(self, x: float, y: float, info) -> tuple[int, int]:
        """
        _summary_

        Args:
            x (float): _description_
            y (float): _description_
            info (_type_): _description_

        Returns:
            tuple[int, int]: _description_
        """
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        return row, col

    def _cell_to_world(self, row: int, col: int, info) -> tuple[float, float]:
        """
        _summary_

        Args:
            row (int): _description_
            col (int): _description_
            info (_type_): _description_

        Returns:
            tuple[float, float]: _description_
        """
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def _replan(self):
        """
        Generates a path by overlaying live dynamic obstacles (Local Costmap) 
        on top of the pre-baked static map (Global Costmap).
        """
        # ----------------------------------------------------
        # 1: Get latest Map, Odom, and Scan Data
        # ----------------------------------------------------
        if self.global_costmap is None or self.robot_xy is None:
            return

        info = self.global_map_info
        cost_map = self.global_costmap.copy()

        # ----------------------------------------------------
        # 2: Inject Local Costmap
        # ----------------------------------------------------
        if self.latest_scan is not None:
            robot_x, robot_y = self.robot_xy
            ranges = np.array(self.latest_scan.ranges)
            angle_min = self.latest_scan.angle_min
            angle_inc = self.latest_scan.angle_increment

            # Map laser hits to grid cells
            hit_cells = set()
            for i, r in enumerate(ranges):
                if np.isnan(r) or r < self.local_min_range or r > self.local_max_range:
                    continue

                beam_angle = self.robot_theta + angle_min + i * angle_inc
                ox, oy = robot_x + r * np.cos(beam_angle), robot_y + r * np.sin(beam_angle)
                
                col = int((ox - info.origin.position.x) / info.resolution)
                row = int((oy - info.origin.position.y) / info.resolution)

                if 0 <= row < info.height and 0 <= col < info.width:
                    hit_cells.add((row, col))
            
            # Calculate integer boundary for array slicing
            local_bound = int(np.ceil(self.local_inf_radius))

            # Localised inflation around new dynamic hits
            for row, col in hit_cells:
                cost_map[row, col] = np.inf

                r0, r1 = max(0, row - local_bound), min(info.height, row + local_bound + 1)
                c0, c1 = max(0, col - local_bound), min(info.width, col + local_bound + 1)
                
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        if cost_map[rr, cc] == np.inf:
                            continue

                        dist = np.hypot(rr - row, cc - col)

                        if dist <= self.local_inf_radius:
                            penalty = self.local_inf_weight * (1.0 - (dist / self.local_inf_radius))
                            cost_map[rr, cc] = max(cost_map[rr, cc], penalty)

        # ----------------------------------------------------
        # 3: Setup A* Search
        # ----------------------------------------------------
        start = self._world_to_cell(*self.robot_xy, info)
        goal = self._world_to_cell(self.goal_x, self.goal_y, info)

        robot_x, robot_y = self.robot_xy
        dist_to_goal = np.hypot(self.goal_x - robot_x, self.goal_y - robot_y)

        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info('Goal reached! Planner going to sleep.', once=True)
            return
        if not (0 <= start[0] < info.height and 0 <= start[1] < info.width):
            self._log_failure('robot outside map bounds')
            self._publish_empty_path()
            return
        if not (0 <= goal[0] < info.height and 0 <= goal[1] < info.width):
            self._log_failure('goal outside map bounds')
            self._publish_empty_path()
            return

        # unblock start cell if robot spawned inside inflation zone (wall = np.inf)
        if cost_map[start] == np.inf:
            cost_map[start] = 0.0

        # get heuristic weight from parameters
        epsilon : float | None = self.get_parameter('planning.heuristic_weight').value

        # pass combined cost_map into A*
        path_cells = astar_search(cost_map, start, goal, epsilon=epsilon)
        
        if path_cells is None:
            self._log_failure('no path found')
            self._publish_empty_path()
            return

        self.consecutive_failures = 0

        # ----------------------------------------------------
        # 4: Convert and Smooth A* Path
        # ----------------------------------------------------
        world_path = []
        for (r, c) in path_cells:
            wx, wy = self._cell_to_world(r, c, info)
            world_path.append((wx, wy))
            
        # grab smoothing weights from params
        data_w : float | None = self.get_parameter('planning.data_weight').value
        smooth_w : float | None = self.get_parameter('planning.smooth_weight').value

        # smooth the trajectory
        smoothed_world_path = self._smooth_path(world_path, data_weight=data_w, smooth_weight=smooth_w)

        # ----------------------------------------------------
        # 5: Publish Smoothed Nav Path
        # ----------------------------------------------------
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame

        # identity orientation for all waypoints — controller handles heading.
        q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # publish the smoothed points
        for (wx, wy) in smoothed_world_path:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation = q
            path_msg.poses.append(ps)

        if path_msg.poses:
            path_msg.poses[-1].pose.position.x = self.goal_x
            path_msg.poses[-1].pose.position.y = self.goal_y

        self.path_pub.publish(path_msg)

        # ----------------------------------------------------
        # 6: Visualise the Costmap in RViz
        # ----------------------------------------------------
        # convert float gradient to 0-100 integers for RViz
        vis_grid = np.zeros_like(cost_map, dtype=np.int8)
        
        # solid walls = 100
        vis_grid[cost_map == np.inf] = 100
        
        # scale gradient (0.0 to inflation_weight) into RViz values (1 to 99)
        gradient_mask = (cost_map > 0) & (cost_map < np.inf)
        if np.any(gradient_mask):
            max_cost = np.max(cost_map[gradient_mask])
            vis_grid[gradient_mask] = np.clip((cost_map[gradient_mask] / max_cost) * 98 + 1, 1, 99).astype(np.int8)

        costmap_msg = OccupancyGridMsg()
        costmap_msg.header = path_msg.header
        costmap_msg.info = info
        costmap_msg.data = vis_grid.flatten().tolist()
        self.costmap_pub.publish(costmap_msg)



    def _smooth_path(self, path: list[tuple[float, float]], data_weight: float, smooth_weight: float, tolerance: float = 0.01):
        """
        Gradient descent path smoothing.

        Args:
            path (list[tuple[float, float]]): _description_
            data_weight (float): _description_
            smooth_weight (float): _description_
            tolerance (float, optional): _description_. Defaults to 0.001.

        Returns:
            _type_: _description_
        """
        if len(path) <= 2:
            return path
        
        new_path = [list(p) for p in path]
        original_path = [list(p) for p in path]
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1): # Don't move start or goal points
                for j in range(2):            # Iterate over x and y
                    aux = new_path[i][j]
                    
                    # Apply data and smoothing forces
                    new_path[i][j] += data_weight * (original_path[i][j] - new_path[i][j]) + \
                                      smooth_weight * (new_path[i-1][j] + new_path[i+1][j] - 2.0 * new_path[i][j])
                    
                    change += abs(aux - new_path[i][j])
                    
        return [(p[0], p[1]) for p in new_path]

    def _publish_empty_path(self):
        """
        _summary_
        """
        empty = Path()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = self.map_frame
        self.path_pub.publish(empty)

    def _log_failure(self, reason: str):
        """
        _summary_

        Args:
            reason (str): _description_
        """
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
