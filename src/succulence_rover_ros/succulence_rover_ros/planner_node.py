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
from geometry_msgs.msg import Point, PoseStamped, Quaternion
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

            'c_engine': 'cpp',                         # Use C++ optimized inflation and optimization

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
            'costmaps.global.update_window_m': 15.0,

            'costmaps.local.inflation_radius_cells': 10.0,
            'costmaps.local.inflation_weight': 15.0,
            'costmaps.local.max_obstacle_range': 2.0,   # Physical local radius
            'costmaps.local.min_obstacle_range': 0.1,
            'costmaps.local.window_size': 6.0,
            
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

        self.global_inf_radius : float      = float(get_p('costmaps.global.inflation_radius_cells'))
        self.global_inf_weight : float      = float(get_p('costmaps.global.inflation_weight'))
        self.global_update_window : float   = float(get_p('costmaps.global.update_window_m'))
        
        self.local_inf_radius : float   = float(get_p('costmaps.local.inflation_radius_cells'))
        self.local_inf_weight : float   = float(get_p('costmaps.local.inflation_weight'))
        self.local_max_range : float    = float(get_p('costmaps.local.max_obstacle_range'))
        self.local_min_range : float    = float(get_p('costmaps.local.min_obstacle_range'))
        self.local_window_size : float  = float(get_p('costmaps.local.window_size'))

        c_engine_mode : str             = str(get_p('c_engine')).lower()
        if c_engine_mode in ['cpp', 'full', 'limited']:
            self.c_engine = 'cpp'
        else:
            self.c_engine = 'python'

        # Sensors
        self.lidar_x_offset : float     = float(get_p('lidar.x_offset'))
        self.lidar_y_offset : float     = float(get_p('lidar.y_offset'))
        self.lidar_yaw_offset : float   = float(get_p('lidar.yaw_offset'))

        # Goal
        self.goal_x : float         = float(get_p('goal.x'))
        self.goal_y : float         = float(get_p('goal.y'))
        self.goal_tolerance : float = float(get_p('goal.tolerance'))

        # internal state
        self.latest_map: OccupancyGridMsg | None    = None
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
            f'goal: ({self.goal_x:.4f}, {self.goal_y:.4f})')


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
        # Convert ROS byte data to a NumPy grid
        grid = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(info.height, info.width).copy()

        # Call the C++ powered vectorized helper from astar.py
        self.global_costmap = inflate_obstacles(
            'Global', self.c_engine,
            grid, self.global_inf_radius,
            self.occ_threshold, self.unknown_as_obstacle,
            self.global_inf_weight
        )

        # store map and info, for callbacks and visualisation.
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

        info = self.global_map_info

        # Prepare base costmap (Global + Local)
        cost_map = self._prepare_cost_map(info)

        # Compute Path (A* + Smoothing)
        smoothed_path = self._compute_path(cost_map, info)
        
        # Publish results
        if smoothed_path:
            self._publish_path(smoothed_path)
            self._visualize_costmap(cost_map, info)
        else:
            self._publish_empty_path()


    def _prepare_cost_map(self, info: MapMetaData) -> np.ndarray:
        """
        Combines static global costmap with dynamic local scan obstacles.

        Args:
            info (MapMetaData): ROS MapMetaData containing resolution and origin.

        Returns:
            np.ndarray: Combined 2D float array representing the active costmap.
        """
        # Start with global
        if self.costmap_mode in ['both', 'global']:
            cost_map = self.global_costmap.copy()
        else:
            cost_map = np.zeros((info.height, info.width), dtype=np.float32)

        # Inject local dynamic obstacles if needed
        if self.costmap_mode in ['both', 'local'] and self.latest_scan is not None:
            self._inject_local_costmap(cost_map, info)
            
        return cost_map


    def _inject_local_costmap(self, cost_map: np.ndarray, info: MapMetaData):
        """
        Projects laser scans into the costmap using vectorized math and C++ inflation.
        """
        # ----------------------------------------------------
        # 2: Inject Local Costmap (ROI-Limited)
        # ----------------------------------------------------
        # A. Define Local Pixel Box around robot
        res = info.resolution
        win = int(self.local_window_size / res)
        r_idx, c_idx = self._world_to_cell(self.robot_xy[0], self.robot_xy[1], info)
        
        r0, r1 = max(0, r_idx - win), min(info.height, r_idx + win)
        c0, c1 = max(0, c_idx - win), min(info.width, c_idx + win)
        
        # B. Create a blank local (ROI) grid
        local_height, local_width = r1 - r0, c1 - c0
        local_grid = np.zeros((local_height, local_width), dtype=np.int8)
        
        # C. Vectorized Projection of Laser Scan (performance-critical)
        ranges = np.array(self.latest_scan.ranges)
        
        # Create a full array of angles
        angles = self.robot_theta + self.lidar_yaw_offset + self.latest_scan.angle_min + \
                 np.arange(len(ranges)) * self.latest_scan.angle_increment
        
        # Filter by range
        valid = (ranges >= self.local_min_range) & (ranges <= self.local_max_range)
        
        # Project laser scan into the local grid
        lidar_x = self.robot_xy[0] + np.cos(self.robot_theta) * self.lidar_x_offset - np.sin(self.robot_theta) * self.lidar_y_offset
        lidar_y = self.robot_xy[1] + np.sin(self.robot_theta) * self.lidar_x_offset + np.cos(self.robot_theta) * self.lidar_y_offset

        ox = lidar_x + ranges[valid] * np.cos(angles[valid])
        oy = lidar_y + ranges[valid] * np.sin(angles[valid])
        
        # Project to grid (indices relative to origin)
        cols = ((ox - info.origin.position.x) / res).astype(int)
        rows = ((oy - info.origin.position.y) / res).astype(int)
        
        # Offset to ROI box
        grid_rows, grid_cols = rows - r0, cols - c0
        
        # Mask points inside the ROI
        mask = (grid_rows >= 0) & (grid_rows < local_height) & (grid_cols >= 0) & (grid_cols < local_width)
        local_grid[grid_rows[mask], grid_cols[mask]] = 100
        
        # D. Call Standard Loop Helper for the local ROI
        local_inflated = inflate_obstacles(
            'Local', self.c_engine,
            local_grid, self.local_inf_radius, self.occ_threshold,
            False,          # Don't treat unknown as obstacle for live scans
            self.local_inf_weight
        )

        # E. Maximum-Blend: Patch the local costs back into the base map
        cost_map[r0:r1, c0:c1] = np.maximum(cost_map[r0:r1, c0:c1], local_inflated)


    def _compute_path(self, cost_map: np.ndarray, info: MapMetaData) -> list[tuple[float, float]] | None:
        """
        Runs A* search and applies trajectory smoothing.
        Returns a list of smoothed (wx, wy) coordinates.
        """
        # ----------------------------------------------------
        # 3: Setup A* Search
        # ----------------------------------------------------
        start = self._world_to_cell(*self.robot_xy, info)
        goal = self._world_to_cell(self.goal_x, self.goal_y, info)

        robot_x, robot_y = self.robot_xy
        dist_to_goal = np.hypot(self.goal_x - robot_x, self.goal_y - robot_y)

        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info('Goal reached! Planner going to sleep.', once=True)
            return None

        # Check map boundaries
        if not (0 <= start[0] < info.height and 0 <= start[1] < info.width):
            self._log_failure('robot outside map bounds')
            return None
        if not (0 <= goal[0] < info.height and 0 <= goal[1] < info.width):
            self._log_failure('goal outside map bounds')
            return None

        # clear inflation halos around start and goal so A* doesn't get trapped
        self._clear_inf_halo(cost_map, start, self.local_inf_radius)
        self._clear_inf_halo(cost_map, goal, self.local_inf_radius)

        # publish reachable cells for debugging
        self._publish_reachable_debug(cost_map, start, info)

        # pass combined cost_map into A*
        path_cells = astar_search(cost_map, start, goal, epsilon=self.epsilon)
        if path_cells is None:
            self._log_failure('no path found')
            return None

        self.consecutive_failures = 0
        
        # ----------------------------------------------------
        # 4: Convert and Smooth A* Path
        # ----------------------------------------------------
        world_path = [self._cell_to_world(r, c, info) for (r, c) in path_cells]
        
        # smooth the trajectory
        return self._smooth_path(
            world_path,
            cost_map,
            self.data_weight, 
            self.smooth_weight, 
            self.smooth_dist, 
            self.smooth_tol
        )


    def _publish_path(self, smoothed_path: list[tuple[float, float]]):
        """
        Constructs and publishes the Path message.
        """
        # ----------------------------------------------------
        # 5: Publish Smoothed Nav Path
        # ----------------------------------------------------
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame
        
        # identity orientation for all waypoints — controller handles heading.
        q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # publish the smoothed points
        poses_list = []
        for (wx, wy) in smoothed_path:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation = q
            poses_list.append(ps)
        
        if poses_list:
            poses_list[-1].pose.position.x = self.goal_x
            poses_list[-1].pose.position.y = self.goal_y
            
        path_msg.poses = poses_list
        self.path_pub.publish(path_msg)


    def _visualize_costmap(self, cost_map: np.ndarray, info: MapMetaData):
        """
        Visualizes the active costmap for debugging in RViz.
        """
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
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = self.map_frame
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


    def _smooth_path(self, path: list[tuple[float, float]], cost_map: np.ndarray, data_weight: float, smooth_weight: float, goal_smooth_dist: float = 2.0, tolerance: float = 0.01):
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
        # Mathematical safety check to prevent IndexErrors
        if len(path) < 3:
            return list(path)
        
        #  Early exit using actual physical distance (meters), fixing the unit bug
        path_distance = np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        if path_distance <= goal_smooth_dist:
            return list(path)
        
        new_path = [list(p) for p in path]
        original_path = [list(p) for p in path]
        change = tolerance

        iterations = 0
        max_iterations = 1000
        
        while change >= tolerance and iterations < max_iterations:
            change = 0.0
            for i in range(1, len(path) - 1): # Don't move start or goal points

                # Dynamic Tension: Calculate physical distance from current node to the goal
                d_to_goal = np.hypot(original_path[-1][0] - new_path[i][0], original_path[-1][1] - new_path[i][1])
                
                # Relax smoothing tension near the goal to prevent pulling paths into adjacent walls
                curr_smooth = smooth_weight * (d_to_goal / goal_smooth_dist) if d_to_goal < goal_smooth_dist else smooth_weight

                for j in range(2):            # Iterate over x and y
                    aux = new_path[i][j]
                    
                    # Apply data alignment force and dynamic structural smoothness force
                    smoothed_val = aux + data_weight * (original_path[i][j] - aux) + \
                                   curr_smooth * (new_path[i-1][j] + new_path[i+1][j] - 2.0 * aux)
                    
                    # Collision check integration
                    temp_pose = list(new_path[i])
                    temp_pose[j] = smoothed_val
                    
                    # Convert to pixel indices, check safety against costmap
                    mx = int((temp_pose[0] - self.global_map_info.origin.position.x) / self.global_map_info.resolution)
                    my = int((temp_pose[1] - self.global_map_info.origin.position.y) / self.global_map_info.resolution)
                    
                    if 0 <= my < cost_map.shape[0] and 0 <= mx < cost_map.shape[1]:
                        if cost_map[my, mx] != np.inf:
                            # Safe to move
                            new_path[i][j] = smoothed_val
                            change += abs(aux - smoothed_val)
                        else:
                            # Revert to original safe A* layout if it hits an infinite-cost wall
                            new_path[i][j] = original_path[i][j]
                    else:
                        # Out of map bounds, revert
                        new_path[i][j] = original_path[i][j]
            
            iterations += 1
                    
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