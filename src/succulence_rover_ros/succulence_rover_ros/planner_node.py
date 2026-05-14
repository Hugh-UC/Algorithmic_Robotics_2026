"""
Planner ROS2 Node (Implements A*)

Subscribes to the SLAM occupancy grid and the SLAM odometry. On a timer,
runs A* from the robot's current cell to the hardcoded goal cell and
publishes a nav_msgs/Path in the map frame.

Usage: The planner node is launched as part of the mission.launch.py file,
        which starts the whole SLAM + planning + navigation stack together.
        The planner can be configured via ROS parameters, which are set in
        the params_sim.yaml and params_physical.yaml files and loaded by the
        launch file.
"""

import numpy as np
import rclpy
from rclpy.node import Node, Publisher
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg, Odometry, Path, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion
from rcl_interfaces.msg import ParameterDescriptor
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .astar import astar_search, inflate_obstacles


class PlannerNode(Node):
    """ROS2 wrapper around A*. Replans at a fixed rate against the latest SLAM map."""

    def __init__(self):
        """
        Subscribes to SLAM map and odometry, publishes A* path.
        """
        super().__init__('planner_node')

        # all required parameter keys (with default fallbacks)
        param_defaults : dict[str, str | int | float | bool] = {
            # Topics/Frames
            'map_topic': '/succulence/map',
            'odom_topic': '/succulence/slam/odometry',
            'plan_topic': '/succulence/plan',
            'frames.map_frame': 'map',
            'costmap_topic': '/succulence/costmap',
            'scan_topic': '/scan',                      # Physical Lidar

            # Planning
            'planning.replan_period': 0.5,
            'planning.heuristic_weight': 1.0,           # Optimal for small 5x5m pen
            'planning.data_weight': 0.4,
            'planning.smooth_weight': 0.3,
            'planning.goal_smooth_distance': 10.0,      # Disable smoothing for last 25cm
            'planning.smooth_tolerance': 0.01,

            # Costmaps
            'costmaps.mode': 'both',
            'costmaps.occupancy_threshold': 50,
            'costmaps.treat_unknown_as_obstacle': False,

            'costmaps.global.inflation_radius_cells': 12.0,
            'costmaps.global.inflation_weight': 50.0,

            'costmaps.local.inflation_radius_cells': 10.0,
            'costmaps.local.inflation_weight': 15.0,
            'costmaps.local.max_obstacle_range': 2.0,   # Physical local radius
            'costmaps.local.min_obstacle_range': 0.1,
            
            # Sensors
            'lidar.x_offset': 0.0,
            'lidar.y_offset': 0.0,
            'lidar.yaw_offset': 1.570,                  # Physical TB4 Lidar rotation

            # Goal
            'goal.x': 0.45,                             # Physical lab test goal
            'goal.y': 2.35,
            'goal.tolerance': 0.05                      # Must match Navigator goal_tolerance
        }
        
        # declare all parameters, with default
        for name, default_val in param_defaults.items():
            self.declare_parameter(name, default_val)

        # tiny helper to fetch values safely and silence 'pylance'
        def get_p(name: str):
            val = self.get_parameter(name).value
            return val if val is not None else param_defaults[name]

        # Topics/Frames
        map_topic : str     = str(get_p('map_topic'))
        odom_topic : str    = str(get_p('odom_topic'))
        plan_topic : str    = str(get_p('plan_topic'))
        costmap_topic : str = str(get_p('costmap_topic'))
        scan_topic : str    = str(get_p('scan_topic'))

        # Planning
        self.map_frame : str        = str(get_p('frames.map_frame'))
        self.replan_period : float  = float(get_p('planning.replan_period'))
        self.epsilon : float        = float(get_p('planning.heuristic_weight'))
        self.data_weight : float    = float(get_p('planning.data_weight'))
        self.smooth_weight : float  = float(get_p('planning.smooth_weight'))
        self.smooth_dist : float    = float(get_p('planning.goal_smooth_distance'))
        self.smooth_tol : float     = float(get_p('planning.smooth_tolerance'))

        # Costmaps
        self.costmap_mode : str         = str(get_p('costmaps.mode')).lower()
        self.occ_threshold : int        = int(get_p('costmaps.occupancy_threshold'))
        self.unknown_as_obstacle : bool = bool(get_p('costmaps.treat_unknown_as_obstacle'))

        self.global_inf_radius : float  = float(get_p('costmaps.global.inflation_radius_cells'))
        self.global_inf_weight : float  = float(get_p('costmaps.global.inflation_weight'))
        
        self.local_inf_radius : float   = float(get_p('costmaps.local.inflation_radius_cells'))
        self.local_inf_weight : float   = float(get_p('costmaps.local.inflation_weight'))
        self.local_max_range : float    = float(get_p('costmaps.local.max_obstacle_range'))
        self.local_min_range : float    = float(get_p('costmaps.local.min_obstacle_range'))

        # Sensors
        self.lidar_x_offset : float     = float(get_p('lidar.x_offset'))
        self.lidar_y_offset : float     = float(get_p('lidar.y_offset'))
        self.lidar_yaw_offset : float   = float(get_p('lidar.yaw_offset'))

        # Goal
        self.goal_x : float         = float(get_p('goal.x'))
        self.goal_y : float         = float(get_p('goal.y'))
        self.goal_tolerance : float = float(get_p('goal.tolerance'))

        # internal state
        self.global_costmap:  np.ndarray | None     = None
        self.global_map_info : MapMetaData | None   = None
        self.latest_scan : LaserScan | None         = None
        
        self.robot_xy : tuple[float, float] | None  = None
        self.robot_theta : float                    = 0.0
        self.consecutive_failures : int             = 0

        # ROS2 publishers and subscribers
        self.path_pub : Publisher       = self.create_publisher(Path, plan_topic, 10)
        self.costmap_pub : Publisher    = self.create_publisher(OccupancyGridMsg, costmap_topic, 10)
        self.reachable_pub : Publisher  = self.create_publisher(OccupancyGridMsg, plan_topic + '/reachable', 10)

        self.create_subscription(OccupancyGridMsg, map_topic, self._map_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        
        # replan on a timer
        self.create_timer(self.replan_period, self._replan)

        # log initialisation info
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

    def _world_to_cell(self, x: float, y: float, info: MapMetaData) -> tuple[int, int]:
        """
        Converts world coordinates (meters) to grid cell indices.

        Args:
            x (float): World X coordinate.
            y (float): World Y coordinate.
            info (MapMetaData): ROS MapMetaData containing resolution and origin.

        Returns:
            tuple[int, int]: The corresponding (row, col) grid indices.
        """
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        return row, col

    def _cell_to_world(self, row: int, col: int, info: MapMetaData) -> tuple[float, float]:
        """
        Converts grid cell indices to world coordinates (meters).

        Args:
            row (int): Grid row index.
            col (int): Grid column index.
            info (MapMetaData): ROS MapMetaData containing resolution and origin.

        Returns:
            tuple[float, float]: The corresponding (X, Y) world coordinates of the cell center.
        """
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y
    
    def _clear_inf_halo(self, cost_map: np.ndarray, center: tuple[int, int], radius_cells: float):
        """
        Clears solid np.inf walls around a center point so A* doesn't get trapped.
        Demotes the solid wall to a heavy gradient penalty so it remains navigable.

        Args:
            cost_map (np.ndarray): The 2D float array representing the costmap.
            center (tuple[int, int]): The (row, col) target center to clear around.
            radius_cells (float): The radial distance in cells to clear.
        """
        h, w = cost_map.shape
        bound = int(np.ceil(radius_cells))
        r0, r1 = max(0, center[0] - bound), min(h, center[0] + bound + 1)
        c0, c1 = max(0, center[1] - bound), min(w, center[1] + bound + 1)
        
        for r in range(r0, r1):
            for c in range(c0, c1):
                if cost_map[r, c] == np.inf:
                    dist = np.hypot(r - center[0], c - center[1])
                    if dist <= radius_cells:
                        # Demote impassable wall to a high-cost navigable zone
                        cost_map[r, c] = self.global_inf_weight

    def _replan(self):
        """
        Generates a path by overlaying live dynamic obstacles (Local Costmap) 
        on top of the pre-baked static map (Global Costmap), computes A*,
        smooths the path, and publishes to ROS topics.
        """
        # ----------------------------------------------------
        # 1: Get latest Map, Odom, and Scan Data
        # ----------------------------------------------------
        if self.global_costmap is None or self.robot_xy is None or self.global_map_info is None:
            return

        info : MapMetaData = self.global_map_info

        if self.costmap_mode in ['both', 'global'] and self.global_costmap is not None:
            cost_map = self.global_costmap.copy()
        else:
            cost_map = np.zeros((info.height, info.width), dtype=np.float32)

        # ----------------------------------------------------
        # 2: Inject Local Costmap
        # ----------------------------------------------------
        if self.costmap_mode in ['both', 'local'] and self.latest_scan is not None:
            robot_x, robot_y = self.robot_xy

            # Project laser scan into world frame, find obstacle locations in grid
            c_r, s_r = np.cos(self.robot_theta), np.sin(self.robot_theta)
            lidar_x = robot_x + c_r * self.lidar_x_offset - s_r * self.lidar_y_offset
            lidar_y = robot_y + s_r * self.lidar_x_offset + c_r * self.lidar_y_offset

            ranges = np.array(self.latest_scan.ranges)
            angle_min = self.latest_scan.angle_min
            angle_inc = self.latest_scan.angle_increment

            # Map laser hits to grid cells
            hit_cells = set()
            for i, r in enumerate(ranges):
                if np.isnan(r) or r < self.local_min_range or r > self.local_max_range:
                    continue

                beam_angle = self.robot_theta + self.lidar_yaw_offset + angle_min + i * angle_inc
                ox, oy = lidar_x + r * np.cos(beam_angle), lidar_y + r * np.sin(beam_angle)
                
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

        # clear inflation halos around start and goal so A* doesn't get trapped
        self._clear_inf_halo(cost_map, start, self.local_inf_radius)
        self._clear_inf_halo(cost_map, goal, self.local_inf_radius)

        # publish reachable cells for debugging
        self._publish_reachable_debug(cost_map, start, info)

        # pass combined cost_map into A*
        path_cells = astar_search(cost_map, start, goal, epsilon=self.epsilon)
        
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

        # smooth the trajectory
        smoothed_world_path = self._smooth_path(
            world_path,
            data_weight=self.data_weight,
            smooth_weight=self.smooth_weight,
            goal_smooth_dist=self.smooth_dist,
            tolerance=self.smooth_tol
        )

        # ----------------------------------------------------
        # 5: Publish Smoothed Nav Path
        # ----------------------------------------------------
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame

        # identity orientation for all waypoints — controller handles heading.
        q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # publish the smoothed points
        poses_list : list[PoseStamped] = []
        for (wx, wy) in smoothed_world_path:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation = q
            poses_list.append(ps)

        if len(poses_list) > 0:
            poses_list[-1].pose.position.x = self.goal_x
            poses_list[-1].pose.position.y = self.goal_y

        path_msg.poses = poses_list

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

    def _publish_reachable_debug(self, cost_map: np.ndarray, start: tuple[int, int], info):
        """
        Diagnoses 'No Path Found' errors by flood-filling the map to see exactly 
        where the robot gets blocked.
        """
        h, w = cost_map.shape
        start_row, start_col = start

        if not (0 <= start_row < h and 0 <= start_col < w) or cost_map[start_row, start_col] == np.inf:
            return

        # Create a boolean mask of free space (True = free, False = solid wall)
        free_space = cost_map != np.inf
        
        # Label all connected components
        process = ndimage.label(free_space, structure=np.ones((3, 3), dtype=bool))
        labels = process[0]

        
        start_label = int(labels[start])
        reachable = (labels == start_label) if start_label != 0 else np.zeros_like(free_space)

        # Encode: 0 = reachable (light), 100 = solid wall (dark), 50 = unreachable/stranded (gray)
        debug = np.full_like(cost_map, 50, dtype=np.int8)
        debug[reachable] = 0
        debug[~free_space] = 100

        msg = OccupancyGridMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info = info
        msg.data = debug.flatten().tolist()
        self.reachable_pub.publish(msg)

    def _smooth_path(self, path: list[tuple[float, float]], data_weight: float, smooth_weight: float, goal_smooth_dist: float = 2.0, tolerance: float = 0.01):
        """
        Gradient descent path smoothing.

        Args:
            path (list[tuple[float, float]]): The raw list of (X, Y) waypoints from A*.
            data_weight (float): The anchor strength pulling waypoints back to original A* positions.
            smooth_weight (float): The tension strength pulling waypoints into straight lines.
            goal_smooth_dist (float, optional): Path length limit below which smoothing is disabled. Defaults to 2.0.
            tolerance (float, optional): Delta threshold for loop convergence. Defaults to 0.01.

        Returns:
            list[tuple[float, float]]: The smoothed trajectory.
        """
        if len(path) <= goal_smooth_dist:
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
        Publishes an empty path to halt the navigator node when planning fails.
        """
        empty = Path()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = self.map_frame
        self.path_pub.publish(empty)

    def _log_failure(self, reason: str):
        """
        Logs planning failures, throttling output to prevent terminal spam.

        Args:
            reason (str): The string description of the failure cause.
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