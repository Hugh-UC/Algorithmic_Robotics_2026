"""
Pose Graph SLAM Node

This node implements pose graph SLAM using odometry and laser scans. It
builds a pose graph where each node is a keyframe (a robot pose at which
a laser scan was taken) and edges represent relative pose constraints
from odometry and scan matching. The graph is optimised periodically to
produce a globally consistent map and trajectory estimate.
The node also builds an occupancy grid map from the laser scans and
publishes it for use by the planner and visualisation.

Key features:
- Keyframe selection based on translation and rotation thresholds
- Scan matching for relative pose estimation between keyframes
- Pose graph optimisation using non-linear least squares
- Map rebuilding after optimisation to correct for drift and ghost walls
- Configurable parameters for tuning SLAM performance and accuracy

Usage: The SlamNode is instantiated in the mission launch file and runs
        as part of the overall SLAM + A* + Navigator system. It
        subscribes to odometry and laser scans, and publishes the
        occupancy grid map, SLAM-corrected odometry, and the SLAM path
        for visualisation in RViz.

References:
    - Dellaert, "Factor Graphs and GTSAM" (2012) — factor graph formulation of SLAM
    - Olson, "Real-Time Correlative Scan Matching" (2009) — scan matching algorithm
    - Thrun, Burgard, Fox, "Probabilistic Robotics" (2005), Chapters 5-9 — occupancy grid mapping, motion models, SLAM
"""

import numpy as np
import time
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid as OccupancyGridMsg
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion, PoseStamped
from scipy.spatial.transform import Rotation

from . import utils
from .occupancy_grid_mapper import OccupancyGrid
from .scan_matcher import ScanMatcher, scans_from_ranges
from .pose_graph import PoseGraph
from . import graph_optimizer

from .motion_model import compute_motion_covariance


class SlamNode(Node):
    """ROS2 node implementing pose graph SLAM."""

    def __init__(self):
        super().__init__('slam_node')

        param_defaults: dict[str, str | int | float | bool] = {
            'scan_topic': '/scan',                      # Physical raw Lidar
            'odom_topic': '/odom',                      # Physical raw Odometry
            'map_topic': '/succulence/map',
            'slam_odometry_topic': '/succulence/slam/odometry',
            'slam_path_topic': '/succulence/slam/path',

            'slam.keyframe_distance': 0.06,             # High density for physical speed
            'slam.keyframe_angle': 0.05,                # High density for physical slip
            'slam.optimization_interval': 3,
            'slam.num_iterations': 10,
            'slam.map_publish_interval': 0.3,           # Fast refresh for physical safety
            'slam.scan_match_cov_xy': 0.003,
            'slam.scan_match_cov_theta': 0.0015,
            
            'slam.map_match_weight': 0.3,
            'slam.scan_rate_limit': 10.0,               # Matches RPLIDAR-A1 motor limit

            'scan_matcher.search_x': 0.5,
            'scan_matcher.search_y': 0.5,
            'scan_matcher.search_theta': 0.1,
            'scan_matcher.resolution_x': 0.025,
            'scan_matcher.resolution_y': 0.025,
            'scan_matcher.resolution_theta': 0.02,
            'scan_matcher.min_score': 0.45,
            'scan_matcher.local_grid_size': 1000,
            'scan_matcher.local_grid_resolution': 0.025,

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

        scan_topic : str        = str(get_p('scan_topic'))
        odom_topic : str        = str(get_p('odom_topic'))
        map_topic : str         = str(get_p('map_topic'))
        slam_odom_topic : str   = str(get_p('slam_odometry_topic'))
        slam_path_topic : str   = str(get_p('slam_path_topic'))

        self.keyframe_distance      = float(get_p('slam.keyframe_distance'))
        self.keyframe_angle         = float(get_p('slam.keyframe_angle'))
        self.optimization_interval  = int(get_p('slam.optimization_interval'))
        self.num_iterations         = int(get_p('slam.num_iterations'))
        map_publish_interval        = float(get_p('slam.map_publish_interval'))
        self.scan_match_cov_xy      = float(get_p('slam.scan_match_cov_xy'))
        self.scan_match_cov_theta   = float(get_p('slam.scan_match_cov_theta'))
        # Map-Scan Weighting
        self.map_match_weight       = float(get_p('slam.map_match_weight'))
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
            local_grid_size=int(get_p('scan_matcher.local_grid_size')),
            local_grid_resolution=float(get_p('scan_matcher.local_grid_resolution')),
            min_score=float(get_p('scan_matcher.min_score')),
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
        self.keyframe_scans: List[Tuple[np.ndarray, float, float]] = []
        self.keyframe_count = 0

        self.last_scan_time = 0.0

        self.map_pub = self.create_publisher(OccupancyGridMsg, map_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, slam_odom_topic, 10)
        self.path_pub = self.create_publisher(Path, slam_path_topic, 10)

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)

        self.map_timer = self.create_timer(map_publish_interval, self.publish_map)

        self.get_logger().info(f'SlamNode started — scan: {scan_topic}, odom: {odom_topic}')
        self.get_logger().info(f'  Keyframes: {self.keyframe_distance}m / {self.keyframe_angle}rad')
        self.get_logger().info(f'  Optimise every {self.optimization_interval} keyframes')

    def odom_callback(self, msg: Odometry):
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

    def scan_callback(self, msg: LaserScan):
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
        """Return True once the robot has translated or rotated past the thresholds."""
        dx = current_pose[0] - last_keyframe_pose[0]
        dy = current_pose[1] - last_keyframe_pose[1]
        dist = np.sqrt(dx * dx + dy * dy)
        angle = abs(utils.normalize_angle(current_pose[2] - last_keyframe_pose[2]))
        return dist > self.keyframe_distance or angle > self.keyframe_angle

    def _process_keyframe(self, scan_msg: LaserScan):
        """Core SLAM loop: add node, add edges (odom + scan-match + map-match), optimise."""
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

            # 2. Absolute Scan-to-Map Matching (YOUR EXTENSION)
            map_matched_global, map_match_cov, map_match_score = self.scan_matcher.match_to_map(
                global_grid=self.occupancy_grid.grid,
                map_origin_x=self.occupancy_grid.origin_x,
                map_origin_y=self.occupancy_grid.origin_y,
                map_resolution=self.occupancy_grid.resolution,
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

            # Add Scan-to-Scan Edge
            if match_score > self.scan_matcher.min_score and not saturated:
                match_cov[0, 0] = max(match_cov[0, 0], self.scan_match_cov_xy)
                match_cov[1, 1] = max(match_cov[1, 1], self.scan_match_cov_xy)
                match_cov[2, 2] = max(match_cov[2, 2], self.scan_match_cov_theta)
                
                # Scale covariance by relative weight (lower weight = higher covariance/less trust)
                match_cov /= (1.0 - self.map_match_weight + 1e-6)
                
                self.pose_graph.add_edge(node_id - 1, node_id, matched_pose, match_cov)
                self.get_logger().info(
                    f'  Scan-match accepted: score={match_score:.3f}, '
                    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}]')
            elif saturated:
                self.get_logger().warn(
                    f'  Scan-match SATURATED (boundary): score={match_score:.3f}, '
                    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}] '
                    f'-- weakened odom edge, no scan-match edge')

            # Add Scan-to-Map Edge (YOUR EXTENSION)
            if map_match_score > self.scan_matcher.min_score:
                map_match_cov[0, 0] = max(map_match_cov[0, 0], self.scan_match_cov_xy)
                map_match_cov[1, 1] = max(map_match_cov[1, 1], self.scan_match_cov_xy)
                map_match_cov[2, 2] = max(map_match_cov[2, 2], self.scan_match_cov_theta)
                
                # Scale covariance by absolute map weight
                map_match_cov /= (self.map_match_weight + 1e-6)
                
                # Absolute constraint: Edge from origin (Node 0) to current node
                self.pose_graph.add_edge(0, node_id, map_matched_global, map_match_cov)
                self.get_logger().info(f'  Map-match accepted: score={map_match_score:.3f}')

        self.last_keyframe_pose = self.current_odom_pose.copy()
        self.last_keyframe_scan = scan_points
        self.current_odom_cov = np.zeros((3, 3))

        self.keyframe_scans.append((
            ranges.copy(), scan_msg.angle_min, angle_increment))
        self.keyframe_count += 1

        if (self.keyframe_count > 1
                and self.keyframe_count % self.optimization_interval == 0):
            self._optimize_and_rebuild()
        else:
            latest_pose = self.pose_graph.nodes[-1]
            self.occupancy_grid.update(
                pose=latest_pose,
                ranges=ranges,
                angle_min=scan_msg.angle_min,
                angle_increment=angle_increment)

        self._publish_path()

        if self.keyframe_count % 5 == 0:
            self.get_logger().info(
                f'Keyframe {self.keyframe_count}: '
                f'{self.pose_graph.get_num_nodes()} nodes, '
                f'{self.pose_graph.get_num_edges()} edges')

    def _optimize_and_rebuild(self):
        self.get_logger().info(
            f'Optimising ({self.pose_graph.get_num_nodes()} nodes, '
            f'{self.pose_graph.get_num_edges()} edges)...')

        graph_optimizer.optimize(self.pose_graph, self.num_iterations)

        optimized_last = self.pose_graph.nodes[-1].copy()
        self.last_keyframe_pose = optimized_last
        self.current_odom_pose = optimized_last

        self._rebuild_map()
        self.get_logger().info('Optimisation complete.')

    def _rebuild_map(self):
        t0 = time.monotonic()
        self.occupancy_grid.grid = np.zeros_like(self.occupancy_grid.grid)
        poses = self.pose_graph.get_poses()
        n_rendered = 0
        for i, (ranges, angle_min, angle_increment) in enumerate(self.keyframe_scans):
            if i >= len(poses):
                break
            self.occupancy_grid.update(
                pose=poses[i], ranges=ranges,
                angle_min=angle_min, angle_increment=angle_increment)
            n_rendered += 1
        n_known = int(np.sum(np.abs(self.occupancy_grid.grid) > 0.1))
        self.get_logger().info(
            f'  Map rebuilt: {n_rendered} keyframes rendered, '
            f'{n_known} known cells, {time.monotonic() - t0:.2f}s')

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
        for pose in self.pose_graph.get_poses():
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = pose[0]
            ps.pose.position.y = pose[1]
            ps.pose.orientation = self._yaw_to_quaternion(pose[2])
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def publish_map(self):
        if self.keyframe_count == 0:
            return
        t0 = time.monotonic()
        map_msg = self.occupancy_grid.to_ros_message(
            frame_id='map', timestamp=self.get_clock().now().to_msg())
        self.map_pub.publish(map_msg)
        n_known = int(np.sum(np.abs(self.occupancy_grid.grid) > 0.1))
        self.get_logger().info(
            f'  Map published: {n_known} known cells '
            f'({100.0*n_known/self.occupancy_grid.grid.size:.2f}% of grid), '
            f'{time.monotonic() - t0:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
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