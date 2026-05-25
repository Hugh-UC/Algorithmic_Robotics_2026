"""
SLAM Estimator Node (The Brain)
Handles Scan Matching, Pose Graph Management, and Optimization.
Publishes the optimized trajectory / path.
"""
import numpy as np
import time
import multiprocessing
import copy
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid as OccupancyGridMsg
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion, PoseStamped
from std_msgs.msg import Int32
from scipy.spatial.transform import Rotation

from . import utils
from .occupancy_grid_mapper import OccupancyGrid
from .scan_matcher import ScanMatcher, scans_from_ranges
from .pose_graph import PoseGraph
from . import graph_optimizer

from .motion_model import compute_motion_covariance

class SlamEstimatorNode(Node):
    def __init__(self):
        super().__init__('slam_estimator')

        param_defaults: dict[str, str | int | float | bool] = {
            'scan_topic': '/scan',                      # Physical raw Lidar
            'odom_topic': '/odom',                      # Physical raw Odometry
            'map_topic': '/succulence/map',
            'slam_odometry_topic': '/succulence/slam/odometry',
            'slam_path_topic': '/succulence/slam/path',
            'keyframe_scan_topic': '/succulence/slam/keyframe_scan',
            'map_version_topic': '/succulence/slam/version',

            'slam.keyframe_distance': 0.06,             # High density for physical speed
            'slam.keyframe_angle': 0.05,                # High density for physical slip
            'slam.optimization_interval': 3,
            'slam.num_iterations': 10,
            'slam.scan_match_cov_xy': 0.003,
            'slam.scan_match_cov_theta': 0.0015,
            
            'slam.map_match_weight': 0.3,
            'slam.maturity_threshold': 0.5,
            'slam.scan_rate_limit': 10.0,               # Matches RPLIDAR-A1 motor limit

            'scan_matcher.search_x': 0.5,
            'scan_matcher.search_y': 0.5,
            'scan_matcher.search_theta': 0.1,
            'scan_matcher.resolution_x': 0.025,
            'scan_matcher.resolution_y': 0.025,
            'scan_matcher.resolution_theta': 0.02,
            'scan_matcher.dilation_size': 3,
            'scan_matcher.coarse_search_multiplier': 4.0,
            'scan_matcher.min_score': 0.45,
            'scan_matcher.local_grid_size': 1000,
            'scan_matcher.local_grid_resolution': 0.025,
            'scan_matcher.edge_trim_degrees': 1.5,
            'scan_matcher.edge_buffer_degrees': 3.0,
            'scan_matcher.edge_min_weight': 0.1,

            'occupancy_grid.resolution': 0.025,
            'occupancy_grid.width': 300,                # Physical 7.5m map
            'occupancy_grid.height': 300,
            'occupancy_grid.origin_x': -1.0,
            'occupancy_grid.origin_y': -1.0,
            'occupancy_grid.log_odds_occupied': 0.85,
            'occupancy_grid.log_odds_free': 0.4,
            'occupancy_grid.log_odds_max': 5.0,
            'occupancy_grid.log_odds_min': -5.0,
            'occupancy_grid.max_range': 7.5,
            'occupancy_grid.min_range': 0.1,
            'occupancy_grid.edge_trim_degrees': 1.5,
            'occupancy_grid.edge_buffer_degrees': 3.0,
            'occupancy_grid.edge_min_weight': 0.1,

            'motion_model.alpha1': 0.1,
            'motion_model.alpha2': 0.05,
            'motion_model.alpha3': 0.05,
            'motion_model.alpha4': 0.3,

            'lidar.x_offset': 0.0,
            'lidar.y_offset': 0.0,
            'lidar.yaw_offset': 1.570                   # Physical Lidar Mount
        }

        # declare all parameters, with default
        for name, default_val in param_defaults.items():
            self.declare_parameter(name, default_val)

        # tiny helper to fetch values safely and silence 'pylance'
        def get_p(name: str):
            val = self.get_parameter(name).value
            return val if val is not None else param_defaults[name]

        scan_topic : str            = str(get_p('scan_topic'))
        odom_topic : str            = str(get_p('odom_topic'))
        map_topic : str             = str(get_p('map_topic'))
        slam_odom_topic : str       = str(get_p('slam_odometry_topic'))
        slam_path_topic : str       = str(get_p('slam_path_topic'))
        self.keyframe_scan_topic    = str(get_p('keyframe_scan_topic'))         # Define topic for filtered keyframe scans
        self.map_version_topic      = str(get_p('map_version_topic'))           # Define topic for map versioning


        self.keyframe_distance      = float(get_p('slam.keyframe_distance'))
        self.keyframe_angle         = float(get_p('slam.keyframe_angle'))
        self.optimization_interval  = int(get_p('slam.optimization_interval'))
        self.num_iterations         = int(get_p('slam.num_iterations'))
        self.scan_match_cov_xy      = float(get_p('slam.scan_match_cov_xy'))
        self.scan_match_cov_theta   = float(get_p('slam.scan_match_cov_theta'))
        # Map-Scan Weighting
        self.map_match_weight       = float(get_p('slam.map_match_weight'))
        self.maturity_threshold     = float(get_p('slam.maturity_threshold'))
        self.scan_rate_limit        = float(get_p('slam.scan_rate_limit'))

        self.alpha1 = float(get_p('motion_model.alpha1'))
        self.alpha2 = float(get_p('motion_model.alpha2'))
        self.alpha3 = float(get_p('motion_model.alpha3'))
        self.alpha4 = float(get_p('motion_model.alpha4'))

        self.lidar_yaw_offset = float(get_p('lidar.yaw_offset'))

        self.scan_matcher = ScanMatcher(
            search_x=float(get_p('scan_matcher.search_x')),
            search_y=float(get_p('scan_matcher.search_y')),
            search_theta=float(get_p('scan_matcher.search_theta')),
            resolution_x=float(get_p('scan_matcher.resolution_x')),
            resolution_y=float(get_p('scan_matcher.resolution_y')),
            resolution_theta=float(get_p('scan_matcher.resolution_theta')),
            dilation_size=int(get_p('scan_matcher.dilation_size')),
            coarse_search_multiplier=float(get_p('scan_matcher.coarse_search_multiplier')),
            local_grid_size=int(get_p('scan_matcher.local_grid_size')),
            local_grid_resolution=float(get_p('scan_matcher.local_grid_resolution')),
            min_score=float(get_p('scan_matcher.min_score')),
            edge_trim_degrees=float(get_p('scan_matcher.edge_trim_degrees')),
            edge_buffer_degrees=float(get_p('scan_matcher.edge_buffer_degrees')),
            edge_min_weight=float(get_p('scan_matcher.edge_min_weight')),
            lidar_yaw_offset=float(get_p('lidar.yaw_offset')),
        )

        self.occupancy_grid = OccupancyGrid(
            resolution=float(get_p('occupancy_grid.resolution')),
            width=int(get_p('occupancy_grid.width')),
            height=int(get_p('occupancy_grid.height')),
            origin_x=float(get_p('occupancy_grid.origin_x')),
            origin_y=float(get_p('occupancy_grid.origin_y')),
            log_odds_occupied=float(get_p('occupancy_grid.log_odds_occupied')),
            log_odds_free=float(get_p('occupancy_grid.log_odds_free')),
            log_odds_max=float(get_p('occupancy_grid.log_odds_max')),
            log_odds_min=float(get_p('occupancy_grid.log_odds_min')),
            max_range=float(get_p('occupancy_grid.max_range')),
            min_range=float(get_p('occupancy_grid.min_range')),
            edge_trim_degrees=float(get_p('occupancy_grid.edge_trim_degrees')),
            edge_buffer_degrees=float(get_p('occupancy_grid.edge_buffer_degrees')),
            edge_min_weight=float(get_p('occupancy_grid.edge_min_weight')),
            lidar_x_offset=float(get_p('lidar.x_offset')),
            lidar_y_offset=float(get_p('lidar.y_offset')),
            lidar_yaw_offset=float(get_p('lidar.yaw_offset')),
        )

        self.pose_graph = PoseGraph()

        self.prev_odom_pose: Optional[np.ndarray] = None
        self.current_odom_pose = np.array([0.0, 0.0, 0.0])
        self.current_odom_cov = np.zeros((3, 3))

        self.last_keyframe_pose: Optional[np.ndarray] = None
        self.last_keyframe_scan: Optional[np.ndarray] = None

        self.keyframe_timestamps = []
        self.keyframe_count = 0

        self.last_scan_time = 0.0

        # map version tracking
        self.optimization_version = 0
        self.active_map_version = 0
        self.waiting_for_map_rebuild = False
        self.map_data_arrived = False

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)

        self.map_sub = self.create_subscription(OccupancyGridMsg, map_topic, self._map_cb, 1)
        self.map_ack_sub = self.create_subscription(Int32, '/succulence/map_ack', self._map_ack_cb, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, slam_odom_topic, 10)
        self.path_pub = self.create_publisher(Path, slam_path_topic, 10)

        self.version_pub = self.create_publisher(Int32, self.map_version_topic, 10)

        self.get_logger().info(f'Slam Estimator started — scan: {scan_topic}, odom: {odom_topic}')
        self.get_logger().info(f'  Keyframes: {self.keyframe_distance}m / {self.keyframe_angle}rad')
        self.get_logger().info(f'  Optimise every {self.optimization_interval} keyframes')

        # Instantiate the publisher for sensor_msgs/msg/LaserScan
        self.keyframe_scan_pub = self.create_publisher(LaserScan, self.keyframe_scan_topic, 10)


    def _map_cb(self, msg: OccupancyGridMsg):
        """
        Receives the global map built by the Global Mapper node.
        Translates ROS probabilistic bytes (-1, 0..100) back to log-odds for Scan Matching.
        """
        raw_data = np.array(msg.data, dtype=np.float32).reshape((msg.info.height, msg.info.width))
        
        grid_log_odds = np.zeros_like(raw_data)
        grid_log_odds[raw_data >= 50] = self.occupancy_grid.log_odds_occ   
        grid_log_odds[(raw_data >= 0) & (raw_data < 50)] = self.occupancy_grid.log_odds_free
        grid_log_odds[raw_data == -1] = 0.0
        
        self.occupancy_grid.grid = grid_log_odds

        self.map_data_arrived = True

        # Check tracking flag condition
        self._check_map_synchronization_status()
    

    def _map_ack_cb(self, msg: Int32):
        """
        Receives confirmation from the Mapper that the map rebuild is complete.
        """
        self.active_map_version = msg.data
        self._check_map_synchronization_status()
    

    def _check_map_synchronization_status(self):
        """
        Safely clears the rebuild wait block only when the active map version 
        fully matches the optimization version and a grid is present.
        """
        if self.optimization_version == self.active_map_version and self.map_data_arrived:
            if self.waiting_for_map_rebuild and self.occupancy_grid.grid is not None:
                self.waiting_for_map_rebuild = False
                self.get_logger().info(
                    f"Map synchronization verified (Version {self.active_map_version}). "
                    "Scan-to-Map matching safely resumed."
                )


    def _odom_cb(self, msg: Odometry):
        odom_pose = self._odom_msg_to_pose(msg)

        if self.prev_odom_pose is None:
            self.prev_odom_pose = odom_pose
            self.current_odom_pose = odom_pose.copy()
            return

        relative_odom = utils.pose_difference(self.prev_odom_pose, odom_pose)

        motion_cov = compute_motion_covariance(
            relative_odom, self.alpha1, self.alpha2, self.alpha3, self.alpha4)

        J1, J2 = utils.pose_compose_jacobians(self.current_odom_pose, relative_odom)
        self.current_odom_pose = utils.pose_compose(self.current_odom_pose, relative_odom)
        self.current_odom_cov = utils.covariance_propagate(
            self.current_odom_cov, motion_cov, J1, J2)

        self.prev_odom_pose = odom_pose
        self._publish_odometry()


    def _scan_cb(self, msg: LaserScan):
        if self.prev_odom_pose is None:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_scan_time < (1.0 / self.scan_rate_limit):
            return
        self.last_scan_time = current_time

        if self.last_keyframe_pose is None:
            self._process_keyframe(msg)
        elif self._should_add_keyframe(self.current_odom_pose, self.last_keyframe_pose):
            self._process_keyframe(msg)


    def _should_add_keyframe(self, current_pose: np.ndarray,
                              last_keyframe_pose: np.ndarray) -> bool:
        """
        Return True once the robot has translated or rotated past the thresholds.

        Args:
            current_pose (np.ndarray): _description_
            last_keyframe_pose (np.ndarray): _description_

        Returns:
            bool: _description_
        """
        dx = current_pose[0] - last_keyframe_pose[0]
        dy = current_pose[1] - last_keyframe_pose[1]
        dist = np.sqrt(dx * dx + dy * dy)
        angle = abs(utils.normalize_angle(current_pose[2] - last_keyframe_pose[2]))
        return dist > self.keyframe_distance or angle > self.keyframe_angle


    def _process_keyframe(self, scan_msg: LaserScan):
        """
        Core SLAM loop: add node, add edges (odom + scan-match + map-match), optimise.

        Args:
            scan_msg (LaserScan): _description_
        """
        # Ensure map exists before matching
        if self.occupancy_grid.grid is None and self.pose_graph.get_num_nodes() > 0:
            return
        
        ranges = np.array(scan_msg.ranges)
        angle_increment = scan_msg.angle_increment
        scan_points = scans_from_ranges(
            ranges, scan_msg.angle_min, angle_increment,
            min_range=self.occupancy_grid.min_range,
            max_range=self.occupancy_grid.max_range,
            lidar_yaw_offset=self.lidar_yaw_offset)

        node_id = self.pose_graph.add_node(self.current_odom_pose)

        if self.last_keyframe_pose is not None and self.last_keyframe_scan is not None:
            odom_relative = utils.pose_difference(
                self.last_keyframe_pose, self.current_odom_pose)

            # 1. Relative Scan-to-Scan Matching
            matched_pose, match_cov, match_score = self.scan_matcher.match(
                self.last_keyframe_scan, scan_points, odom_relative)

            # 2. Absolute Scan-to-Map Matching
            map_matched_global = None
            map_match_score = 0.0

            if self.active_map_version == self.optimization_version and not self.waiting_for_map_rebuild and self.map_data_arrived:
                map_matched_global, map_match_cov, map_match_score = self.scan_matcher.match_to_map(
                    occupancy_grid=self.occupancy_grid,
                    scan_new=scan_points,
                    initial_guess_global=self.current_odom_pose
                )

            # Detect search-window saturation
            sx = self.scan_matcher.search_x
            sy = self.scan_matcher.search_y
            st = self.scan_matcher.search_theta
            shift = matched_pose - odom_relative
            saturated = (abs(shift[0]) >= 0.95 * sx or
                         abs(shift[1]) >= 0.95 * sy or
                         abs(shift[2]) >= 0.95 * st)

            # Odometry edge
            odom_cov = compute_motion_covariance(
                odom_relative, self.alpha1, self.alpha2,
                self.alpha3, self.alpha4)
            for i in range(3):
                odom_cov[i, i] = max(odom_cov[i, i], self.current_odom_cov[i, i], 1e-4)

            self.pose_graph.add_edge(node_id - 1, node_id, odom_relative, odom_cov)

            # map_match_ratio is the % of scan points hitting confirmed obstacles in the map.
            map_match_ratio = map_match_score

            # trust_multiplier ramps trust from 0.0 to 1.0 as ratio approaches maturity_threshold.
            trust_multiplier = min(1.0, map_match_ratio / self.maturity_threshold)
            
            # effective_map_weight is the final blending ratio (e.g., 0.3 * 0.5 maturity = 0.15 trust).
            effective_map_weight = self.map_match_weight * trust_multiplier

            # Add Scan-to-Scan Edge
            if match_score > self.scan_matcher.min_score and not saturated:
                match_cov[0, 0] = max(match_cov[0, 0], self.scan_match_cov_xy)
                match_cov[1, 1] = max(match_cov[1, 1], self.scan_match_cov_xy)
                match_cov[2, 2] = max(match_cov[2, 2], self.scan_match_cov_theta)
                
                # Scale covariance by relative weight (lower weight = higher covariance/less trust)
                match_cov /= (1.0 - effective_map_weight + 1e-6)
                
                self.pose_graph.add_edge(node_id - 1, node_id, matched_pose, match_cov)

                #self.get_logger().info(
                #    f'  Scan-match accepted: score={match_score:.3f}, '
                #    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}]')
                
            elif saturated:
                self.get_logger().warn(
                    f'  Scan-match SATURATED (boundary): score={match_score:.3f}, '
                    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}] '
                    f'-- weakened odom edge, no scan-match edge')

            # Add Scan-to-Map Edge
            if map_match_score > self.scan_matcher.min_score and effective_map_weight > 0.01:
                map_match_cov[0, 0] = max(map_match_cov[0, 0], self.scan_match_cov_xy)
                map_match_cov[1, 1] = max(map_match_cov[1, 1], self.scan_match_cov_xy)
                map_match_cov[2, 2] = max(map_match_cov[2, 2], self.scan_match_cov_theta)
                
                # Scale covariance by absolute map weight
                map_match_cov /= (effective_map_weight + 1e-6)

                # Compute relative measurement from Node 0 to the global map-matched pose
                map_match_measurement = utils.pose_difference(self.pose_graph.nodes[0], map_matched_global)
                
                # Absolute constraint: Edge from origin (Node 0) to current node
                self.pose_graph.add_edge(0, node_id, map_match_measurement, map_match_cov)

                #self.get_logger().info(
                #    f'  Map-match accepted: ratio={map_match_ratio:.2f}, '
                #    f'effective_weight={effective_map_weight:.2f}')

        self.last_keyframe_pose = self.current_odom_pose.copy()
        self.last_keyframe_scan = scan_points
        self.current_odom_cov = np.zeros((3, 3))

        self.keyframe_timestamps.append(scan_msg.header.stamp)
        self.keyframe_count += 1

        # Broadcast raw scan keyframe
        self.keyframe_scan_pub.publish(scan_msg)

        if (self.keyframe_count > 1 and self.keyframe_count % self.optimization_interval == 0):
            self._run_optimizer()

        self._publish_path()

        if self.keyframe_count % self.optimization_interval == 0:
            self.get_logger().info(
                f'Keyframe {self.keyframe_count}: '
                f'{self.pose_graph.get_num_nodes()} nodes, '
                f'{self.pose_graph.get_num_edges()} edges')
            
        # self.get_logger().info(f'🏁 Keyframe Processing Finished at {self.get_clock().now().nanoseconds / 1e9:.2f}')

    def _run_optimizer(self):
        # Prevent spawning multiple optimizers if one is already running
        if hasattr(self, 'optimizing') and self.optimizing:
            return
        
        # Explicitly set flag to force keyframes to bypass map matching until global_mapper refreshes
        self.waiting_for_map_rebuild = True

        self.map_data_arrived = False

        self.optimizing = True
        self.get_logger().info(
            f'Optimising ({self.pose_graph.get_num_nodes()} nodes, '
            f'{self.pose_graph.get_num_edges()} edges)...')

        # Create a communication channel
        self.opt_queue = multiprocessing.Queue()

        p = multiprocessing.Process(target=run_optimization_process, args=(self.pose_graph, self.num_iterations, self.opt_queue))
        p.start()

        # Create a fast, non-blocking timer to check for the result
        self.opt_timer = self.create_timer(0.1, self._check_optimizer_result)

    
    def _check_optimizer_result(self):
        # If background process have put the result in queue, grab it!
        if not self.opt_queue.empty():
            optimized_graph, duration = self.opt_queue.get()

            # Guard
            if optimized_graph.get_num_nodes() == 0:
                self.optimizing = False
                self.opt_timer.cancel()
                return
            
            # Determine size of historical snapshot that was optimised
            num_optimized_nodes = optimized_graph.get_num_nodes()

            # 1. Calculate 'snap' delta (How much did last node move?)
            old_boundary_node = self.pose_graph.nodes[num_optimized_nodes - 1]
            new_boundary_node = optimized_graph.nodes[num_optimized_nodes - 1]
            correction_delta = utils.pose_difference(old_boundary_node, new_boundary_node)

            # 2. Safely swap drifting graph for newly solved graph
            for i in range(num_optimized_nodes):
                self.pose_graph.nodes[i] = optimized_graph.nodes[i].copy()

            # 3. Propagate correction delta to any new keyframes accumulated
            for i in range(num_optimized_nodes, len(self.pose_graph.nodes)):
                self.pose_graph.nodes[i] = utils.pose_compose(self.pose_graph.nodes[i], correction_delta)

            # 4. Safely update the last keyframe tracking pose to the end of the updated live chain
            self.last_keyframe_pose = self.pose_graph.nodes[-1].copy()

            # 5. Apply correction delta to LIVE odometry
            self.current_odom_pose = utils.pose_compose(self.current_odom_pose, correction_delta)
            
            # 6. Publish the new path BEFORE triggering the Mapper's version ACK
            self._publish_path()

            # 7. Increment map version and broadcast
            self.optimization_version += 1
            self.version_pub.publish(Int32(data=self.optimization_version))
            self.get_logger().info(f'🏁 Optimisation merged successfully, took {duration:.2f}s')
            
            # 8. Clean up
            self.optimizing = False
            self.opt_timer.cancel()

    
    def _odom_msg_to_pose(self, msg: Odometry) -> np.ndarray:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        return np.array([x, y, rotation.as_euler('xyz', degrees=False)[2]])


    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Rotation.from_euler('z', yaw).as_quat(canonical=False)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


    def _3x3_to_6x6_covariance(self, cov_3x3: np.ndarray) -> list:
        cov = [0.0] * 36
        cov[0] = cov_3x3[0, 0]
        cov[1] = cov_3x3[0, 1]
        cov[6] = cov_3x3[1, 0]
        cov[7] = cov_3x3[1, 1]
        cov[35] = cov_3x3[2, 2]
        return cov


    def _publish_odometry(self):
        if self.pose_graph.get_num_nodes() > 0:
            last_graph_pose = self.pose_graph.nodes[-1]
            if self.last_keyframe_pose is not None:
                rel = utils.pose_difference(
                    self.last_keyframe_pose, self.current_odom_pose)
                corrected = utils.pose_compose(last_graph_pose, rel)
            else:
                rel = np.zeros(3)
                corrected = last_graph_pose
        else:
            rel = self.current_odom_pose
            corrected = self.current_odom_pose

        # Covariance since the last keyframe -- alpha model applied to the full
        # delta so the ellipse actually grows visibly between keyframes and
        # snaps back when SLAM commits a new keyframe. The propagated
        # current_odom_cov is used as a floor (it carries any per-step
        # residuals).
        published_cov = compute_motion_covariance(
            rel, self.alpha1, self.alpha2, self.alpha3, self.alpha4)
        for i in range(3):
            published_cov[i, i] = max(
                published_cov[i, i], self.current_odom_cov[i, i])

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = corrected[0]
        msg.pose.pose.position.y = corrected[1]
        msg.pose.pose.orientation = self._yaw_to_quaternion(corrected[2])
        msg.pose.covariance = self._3x3_to_6x6_covariance(published_cov)
        self.odom_pub.publish(msg)


    def _publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        # Create a standard Python list to satisfy Pylance/ROS2 typings
        poses_list = []
        for i, pose in enumerate(self.pose_graph.get_poses()):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            
            # Attach historical timestamps to each node for the global mapper
            if i < len(self.keyframe_timestamps):
                ps.header.stamp = self.keyframe_timestamps[i]
            else:
                ps.header.stamp = path_msg.header.stamp
                
            ps.pose.position.x = float(pose[0])
            ps.pose.position.y = float(pose[1])
            ps.pose.orientation = self._yaw_to_quaternion(pose[2])
            poses_list.append(ps)
            
        path_msg.poses = poses_list
        self.path_pub.publish(path_msg)



def run_optimization_process(graph, num_iterations, result_queue):
    """
    Runs isolated from the ROS node to prevent Pickling errors.

    Args:
        graph (_type_): _description_
        num_iterations (_type_): _description_
        result_queue (_type_): _description_
    """
    t0 = time.monotonic()
    
    graph_optimizer.optimize(graph, num_iterations)
    
    result_queue.put((graph, time.monotonic() - t0))



def main(args=None):
    rclpy.init(args=args)
    node = SlamEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down: {node.keyframe_count} keyframes, '
            f'{node.pose_graph.get_num_nodes()} nodes, '
            f'{node.pose_graph.get_num_edges()} edges')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()