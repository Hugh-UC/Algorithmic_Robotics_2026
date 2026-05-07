"""
Pose Graph SLAM Node (Week 7) — SOLUTION
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

        self.declare_parameter('scan_topic')
        self.declare_parameter('odom_topic')
        self.declare_parameter('map_topic')
        self.declare_parameter('slam_odometry_topic')
        self.declare_parameter('slam_path_topic')

        self.declare_parameter('slam.keyframe_distance')
        self.declare_parameter('slam.keyframe_angle')
        self.declare_parameter('slam.optimization_interval')
        self.declare_parameter('slam.num_iterations')
        self.declare_parameter('slam.map_publish_interval')
        self.declare_parameter('slam.scan_match_cov_xy')
        self.declare_parameter('slam.scan_match_cov_theta')

        self.declare_parameter('scan_matcher.search_x')
        self.declare_parameter('scan_matcher.search_y')
        self.declare_parameter('scan_matcher.search_theta')
        self.declare_parameter('scan_matcher.resolution_x')
        self.declare_parameter('scan_matcher.resolution_y')
        self.declare_parameter('scan_matcher.resolution_theta')
        self.declare_parameter('scan_matcher.min_score')
        self.declare_parameter('scan_matcher.local_grid_size')
        self.declare_parameter('scan_matcher.local_grid_resolution')

        self.declare_parameter('occupancy_grid.resolution')
        self.declare_parameter('occupancy_grid.width')
        self.declare_parameter('occupancy_grid.height')
        self.declare_parameter('occupancy_grid.origin_x')
        self.declare_parameter('occupancy_grid.origin_y')
        self.declare_parameter('occupancy_grid.log_odds_occupied')
        self.declare_parameter('occupancy_grid.log_odds_free')
        self.declare_parameter('occupancy_grid.log_odds_max')
        self.declare_parameter('occupancy_grid.log_odds_min')
        self.declare_parameter('occupancy_grid.max_range')
        self.declare_parameter('occupancy_grid.min_range')

        self.declare_parameter('motion_model.alpha1')
        self.declare_parameter('motion_model.alpha2')
        self.declare_parameter('motion_model.alpha3')
        self.declare_parameter('motion_model.alpha4')

        self.declare_parameter('lidar.x_offset')
        self.declare_parameter('lidar.y_offset')
        self.declare_parameter('lidar.yaw_offset')

        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        map_topic = self.get_parameter('map_topic').value
        slam_odom_topic = self.get_parameter('slam_odometry_topic').value
        slam_path_topic = self.get_parameter('slam_path_topic').value

        self.keyframe_distance = self.get_parameter('slam.keyframe_distance').value
        self.keyframe_angle = self.get_parameter('slam.keyframe_angle').value
        self.optimization_interval = self.get_parameter('slam.optimization_interval').value
        self.num_iterations = self.get_parameter('slam.num_iterations').value
        map_publish_interval = self.get_parameter('slam.map_publish_interval').value
        self.scan_match_cov_xy = self.get_parameter('slam.scan_match_cov_xy').value
        self.scan_match_cov_theta = self.get_parameter('slam.scan_match_cov_theta').value

        self.alpha1 = self.get_parameter('motion_model.alpha1').value
        self.alpha2 = self.get_parameter('motion_model.alpha2').value
        self.alpha3 = self.get_parameter('motion_model.alpha3').value
        self.alpha4 = self.get_parameter('motion_model.alpha4').value

        self.lidar_yaw_offset = self.get_parameter('lidar.yaw_offset').value

        self.scan_matcher = ScanMatcher(
            search_x=self.get_parameter('scan_matcher.search_x').value,
            search_y=self.get_parameter('scan_matcher.search_y').value,
            search_theta=self.get_parameter('scan_matcher.search_theta').value,
            resolution_x=self.get_parameter('scan_matcher.resolution_x').value,
            resolution_y=self.get_parameter('scan_matcher.resolution_y').value,
            resolution_theta=self.get_parameter('scan_matcher.resolution_theta').value,
            local_grid_size=self.get_parameter('scan_matcher.local_grid_size').value,
            local_grid_resolution=self.get_parameter('scan_matcher.local_grid_resolution').value,
            min_score=self.get_parameter('scan_matcher.min_score').value,
        )

        self.occupancy_grid = OccupancyGrid(
            resolution=self.get_parameter('occupancy_grid.resolution').value,
            width=self.get_parameter('occupancy_grid.width').value,
            height=self.get_parameter('occupancy_grid.height').value,
            origin_x=self.get_parameter('occupancy_grid.origin_x').value,
            origin_y=self.get_parameter('occupancy_grid.origin_y').value,
            log_odds_occupied=self.get_parameter('occupancy_grid.log_odds_occupied').value,
            log_odds_free=self.get_parameter('occupancy_grid.log_odds_free').value,
            log_odds_max=self.get_parameter('occupancy_grid.log_odds_max').value,
            log_odds_min=self.get_parameter('occupancy_grid.log_odds_min').value,
            max_range=self.get_parameter('occupancy_grid.max_range').value,
            min_range=self.get_parameter('occupancy_grid.min_range').value,
            lidar_x_offset=self.get_parameter('lidar.x_offset').value,
            lidar_y_offset=self.get_parameter('lidar.y_offset').value,
            lidar_yaw_offset=self.get_parameter('lidar.yaw_offset').value,
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
        self.scan_rate_limit = 10.0  # match the sim's 10 Hz scan topic; the per-scan
                                     # keyframe check is cheap, only matched scans cost

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
        """Core SLAM loop: add node, add edges (odom + scan-match), optimise."""
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

            # Run scan-match first so we know whether to trust odom on this
            # keyframe (saturation -> odom is also unreliable).
            matched_pose, match_cov, match_score = self.scan_matcher.match(
                self.last_keyframe_scan, scan_points, odom_relative)

            # Detect search-window saturation: the matcher's peak is sitting
            # within 5% of the search boundary, which means the true optimum
            # is outside the window and we got a clipped, biased answer.
            # Score can still look high (random correlations at the edge),
            # so min_score alone won't catch it. Threshold loosened from 0.85
            # to 0.95 -- 0.85 was rejecting too many borderline-good matches
            # and starving the graph of scan-match edges.
            sx = self.scan_matcher.search_x
            sy = self.scan_matcher.search_y
            st = self.scan_matcher.search_theta
            shift = matched_pose - odom_relative
            saturated = (abs(shift[0]) >= 0.95 * sx or
                         abs(shift[1]) >= 0.95 * sy or
                         abs(shift[2]) >= 0.95 * st)

            # Odometry edge covariance: alpha model on the full delta
            # (delta^2 scaling means accumulating tiny per-message Q's
            # under-counts noise by ~100x). When scan-match saturates we
            # keep odom at full strength -- the keyframe's only anchor.
            odom_cov = compute_motion_covariance(
                odom_relative, self.alpha1, self.alpha2,
                self.alpha3, self.alpha4)
            for i in range(3):
                odom_cov[i, i] = max(odom_cov[i, i],
                                     self.current_odom_cov[i, i],
                                     1e-4)

            self.pose_graph.add_edge(node_id - 1, node_id, odom_relative, odom_cov)

            if match_score > self.scan_matcher.min_score and not saturated:
                match_cov[0, 0] = max(match_cov[0, 0], self.scan_match_cov_xy)
                match_cov[1, 1] = max(match_cov[1, 1], self.scan_match_cov_xy)
                match_cov[2, 2] = max(match_cov[2, 2], self.scan_match_cov_theta)
                self.pose_graph.add_edge(node_id - 1, node_id, matched_pose, match_cov)
                self.get_logger().info(
                    f'  Scan-match accepted: score={match_score:.3f}, '
                    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}]')
            elif saturated:
                self.get_logger().warn(
                    f'  Scan-match SATURATED (boundary): score={match_score:.3f}, '
                    f'shift=[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}] '
                    f'-- weakened odom edge, no scan-match edge')
            else:
                self.get_logger().warn(
                    f'  Scan-match REJECTED low score: score={match_score:.3f} < '
                    f'min_score={self.scan_matcher.min_score}')

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
        q = Rotation.from_euler('z', yaw).as_quat()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()