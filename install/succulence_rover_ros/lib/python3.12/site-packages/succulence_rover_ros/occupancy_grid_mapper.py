"""
Bayesian Occupancy Grid Mapping (Week 6)

This module builds a 2D occupancy grid map from laser scans and robot poses.
It uses log-odds representation for numerically stable Bayesian updates,
and Bresenham's algorithm for ray-tracing through the grid.

When driven by dead-reckoning poses (from the motion model), the map will
develop "ghost walls" — the same physical wall appearing twice because the
robot's pose estimate was wrong on the second pass. These ghost walls are
the motivation for scan matching (Week 7) and SLAM (Week 8).

The occupancy grid theory (log-odds, Bresenham ray-tracing) will be covered
in the Week 6 lecture. After that lecture, revisit this file and trace
through _ray_trace() and update() to understand how the map is built.

References:
  - Thrun, Burgard, Fox, "Probabilistic Robotics" (2005), Chapter 9
  - Lecture 06: Occupancy Grids and Scan Matching
"""

import array
import numpy as np
from typing import Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation


# ============================================================================
# Helper functions
# ============================================================================

def probability_to_log_odds(p: float) -> float:
    """Convert probability to log-odds: L = log(p / (1-p))."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def log_odds_to_probability(l: float) -> float:
    """Convert log-odds to probability: p = 1 / (1 + exp(-L))."""
    return 1.0 / (1.0 + np.exp(-l))


# ============================================================================
# OccupancyGrid — the core algorithm class
# ============================================================================

class OccupancyGrid:
    """
    2D occupancy grid using Bayesian log-odds updates.

    Log-odds representation:
      - L = 0  means unknown (50% probability)
      - L > 0  means likely occupied
      - L < 0  means likely free
      - Updates are additive: L_new = L_old + L_update
    """

    def __init__(self,
                 resolution: float,
                 width: int,
                 height: int,
                 origin_x: float,
                 origin_y: float,
                 log_odds_occupied: float,
                 log_odds_free: float,
                 log_odds_max: float,
                 log_odds_min: float,
                 max_range: float,
                 min_range: float,
                 lidar_x_offset: float,
                 lidar_y_offset: float,
                 lidar_yaw_offset: float):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y

        # Convert probability parameters to log-odds for the update rule
        self.log_odds_occ = probability_to_log_odds(log_odds_occupied)
        self.log_odds_free = probability_to_log_odds(log_odds_free)
        self.log_odds_max = log_odds_max
        self.log_odds_min = log_odds_min

        self.max_range = max_range
        self.min_range = min_range

        # Lidar mounting offset relative to base_link
        # Set these to match your robot's TF: base_link → lidar_link
        # Default (0, 0, 0) = lidar is at the same position/orientation as base_link
        self.lidar_x_offset = lidar_x_offset
        self.lidar_y_offset = lidar_y_offset
        self.lidar_yaw_offset = lidar_yaw_offset

        # Grid initialised to zero log-odds (= 50% probability = unknown)
        self.grid = np.zeros((height, width), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (metres) to grid coordinates (row, col).

        Args:
            x: World x position in metres
            y: World y position in metres

        Returns:
            (row, col) tuple — integer grid cell indices

        Hints:
            - Subtract the grid origin, then divide by the resolution
            - Cast to int (grid indices must be integers)
            - col corresponds to x, row corresponds to y
        """
        # TODO: YOUR CODE HERE (2 lines)
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cell centre)."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def is_valid_cell(self, row: int, col: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    # ========================================================================
    # Bresenham's Line Algorithm (see Week 6 lecture for theory)
    # ========================================================================

    def _ray_trace(self, start: Tuple[int, int], end: Tuple[int, int]) -> list:
        """
        Trace a line from start to end using Bresenham's algorithm.
        Returns all cells along the ray EXCEPT the endpoint.

        The endpoint is handled separately — it gets the "occupied" update,
        while all cells returned here get the "free" update.

        Args:
            start: Starting grid cell (row, col) — the robot position
            end:   Ending grid cell (row, col) — where the laser beam hit

        Returns:
            List of (row, col) tuples for FREE cells along the ray

        Algorithm (Bresenham):
            1. drow = abs(row1 - row0), dcol = abs(col1 - col0)
            2. srow = sign(row1 - row0), scol = sign(col1 - col0)
            3. err = dcol - drow
            4. Loop:
               a. If at endpoint → break (don't include it)
               b. Append current (row, col) to cells
               c. e2 = 2 * err
               d. If e2 > -drow: err -= drow, col += scol
               e. If e2 <  dcol: err += dcol, row += srow
            5. Return cells
        """
        row0, col0 = start
        row1, col1 = end

        cells = []

        drow = abs(row1 - row0)
        dcol = abs(col1 - col0)

        srow = 1 if row1 > row0 else -1
        scol = 1 if col1 > col0 else -1

        err = dcol - drow

        row, col = row0, col0

        while True:
            # Don't include endpoint
            if row == row1 and col == col1:
                break

            cells.append((row, col))

            e2 = 2 * err

            if e2 > -drow:
                err -= drow
                col += scol

            if e2 < dcol:
                err += dcol
                row += srow

        return cells

    # ========================================================================
    # Bayesian Occupancy Grid Update (see Week 6 lecture for theory)
    # ========================================================================

    def update(self, pose: np.ndarray, ranges: np.ndarray,
               angle_min: float, angle_increment: float):
        """
        Update occupancy grid with a laser scan.

        For each laser beam:
          1. Compute where the beam hit in the world
          2. Ray-trace from the robot to the hit point
          3. Mark cells along the ray as FREE (decrease log-odds)
          4. Mark the endpoint as OCCUPIED (increase log-odds)

        Args:
            pose:            Robot pose [x, y, theta] in world frame
            ranges:          Laser scan ranges in metres
            angle_min:       Start angle of the scan (radians)
            angle_increment: Angular step between beams (radians)
        """
        robot_x, robot_y, robot_theta = pose

        # Compute lidar position in world frame (apply mounting offset)
        c_r = np.cos(robot_theta)
        s_r = np.sin(robot_theta)
        lidar_x = robot_x + c_r * self.lidar_x_offset - s_r * self.lidar_y_offset
        lidar_y = robot_y + s_r * self.lidar_x_offset + c_r * self.lidar_y_offset

        # Lidar position in grid coordinates
        robot_row, robot_col = self.world_to_grid(lidar_x, lidar_y)

        if not self.is_valid_cell(robot_row, robot_col):
            return  # Robot outside grid

        # Process each laser beam
        for i, r in enumerate(ranges):
            # Skip invalid measurements
            if np.isnan(r) or r < self.min_range or r > self.max_range:
                continue

            # Compute beam angle in world frame (including lidar yaw offset)
            beam_angle = robot_theta + self.lidar_yaw_offset + (angle_min + i * angle_increment)

            # TODO: Compute endpoint in world coordinates (2 lines)
            # The beam travels distance r from (lidar_x, lidar_y) along beam_angle.
            end_x = lidar_x + r * np.cos(beam_angle)
            end_y = lidar_y + r * np.sin(beam_angle)

            # Convert to grid coordinates
            end_row, end_col = self.world_to_grid(end_x, end_y)

            if not self.is_valid_cell(end_row, end_col):
                continue  # Endpoint outside grid

            # Trace ray from robot to endpoint
            free_cells = self._ray_trace((robot_row, robot_col), (end_row, end_col))

            # Update free space cells
            for (row, col) in free_cells:
                if self.is_valid_cell(row, col):
                    self.grid[row, col] += self.log_odds_free
                    self.grid[row, col] = max(self.grid[row, col], self.log_odds_min)

            # Update occupied endpoint
            self.grid[end_row, end_col] += self.log_odds_occ
            self.grid[end_row, end_col] = min(self.grid[end_row, end_col], self.log_odds_max)

    # ========================================================================
    # Provided helper methods (do not modify)
    # ========================================================================

    def get_probability_grid(self) -> np.ndarray:
        """Convert log-odds grid to probability grid [0, 1]."""
        return log_odds_to_probability(self.grid)

    def get_ros_occupancy_grid(self) -> np.ndarray:
        """Convert to ROS format: -1 = unknown, 0 = free, 100 = occupied."""
        occupancy = np.zeros_like(self.grid, dtype=np.int8)
        unknown_mask = np.abs(self.grid) < 0.1
        occupancy[unknown_mask] = -1
        known_mask = ~unknown_mask
        prob = 1.0 / (1.0 + np.exp(-self.grid[known_mask]))
        occupancy[known_mask] = (prob * 100).astype(np.int8)
        return occupancy

    def to_ros_message(self, frame_id: str = 'map', timestamp=None) -> OccupancyGridMsg:
        """Convert to nav_msgs/OccupancyGrid message for RViz."""
        msg = OccupancyGridMsg()
        msg.header = Header()
        msg.header.frame_id = frame_id
        if timestamp is not None:
            msg.header.stamp = timestamp
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        occupancy = self.get_ros_occupancy_grid()
        msg.data = array.array('b', occupancy.ravel().tobytes())
        return msg


# ============================================================================
# OccupancyGridMapperNode — the ROS2 node (provided, do not modify)
# ============================================================================

class OccupancyGridMapperNode(Node):
    """
    ROS2 node that builds an occupancy grid from laser scans and odometry.

    Subscribes to:
      - Laser scans (from the robot's lidar)
      - Odometry (dead-reckoning pose from the motion model)

    Publishes:
      - /succulence/map/odom_only (OccupancyGrid)
    """

    def __init__(self):
        super().__init__('occupancy_grid_mapper')

        # --- Parameters (all values come from params.yaml) ---
        self.declare_parameter('scan_topic')
        self.declare_parameter('odom_topic')
        self.declare_parameter('map_topic')
        self.declare_parameter('map_publish_rate')

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

        self.declare_parameter('lidar.x_offset')
        self.declare_parameter('lidar.y_offset')
        self.declare_parameter('lidar.yaw_offset')

        # --- Read parameters ---
        scan_topic = self.get_parameter('scan_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        map_topic = self.get_parameter('map_topic').value
        publish_rate = self.get_parameter('map_publish_rate').value

        # --- Create occupancy grid ---
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

        # --- State ---
        self.current_pose = None
        self.scan_count = 0
        self.last_scan_time = 0.0
        self.scan_rate_limit = 5.0  # Max scans processed per second

        # --- Publishers / Subscribers ---
        self.map_pub = self.create_publisher(OccupancyGridMsg, map_topic, 10)

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)

        # Publish map at fixed rate
        self.map_timer = self.create_timer(1.0 / publish_rate, self.publish_map)

        self.get_logger().info(f'OccupancyGridMapper started')
        self.get_logger().info(f'  Scans: {scan_topic}')
        self.get_logger().info(f'  Odometry: {odom_topic}')
        self.get_logger().info(f'  Map: {map_topic}')
        self.get_logger().info(
            f'  Grid: {self.occupancy_grid.width}x{self.occupancy_grid.height} '
            f'@ {self.occupancy_grid.resolution}m/cell')

    def odom_callback(self, msg: Odometry):
        """Update current pose from dead-reckoning odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        theta = rotation.as_euler('xyz', degrees=False)[2]
        self.current_pose = np.array([x, y, theta])

    def scan_callback(self, msg: LaserScan):
        """Process laser scan: update occupancy grid."""
        if self.current_pose is None:
            return  # No pose yet

        # Rate-limit scan processing
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_scan_time < (1.0 / self.scan_rate_limit):
            return
        self.last_scan_time = current_time

        # Update the grid using Bayesian log-odds (see _ray_trace and update)
        ranges = np.array(msg.ranges)
        self.occupancy_grid.update(
            pose=self.current_pose,
            ranges=ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment
        )
        self.scan_count += 1

        if self.scan_count % 20 == 0:
            self.get_logger().info(f'Scans processed: {self.scan_count}')

    def publish_map(self):
        """Publish occupancy grid as ROS message."""
        if self.scan_count == 0:
            return
        map_msg = self.occupancy_grid.to_ros_message(
            frame_id='map',
            timestamp=self.get_clock().now().to_msg()
        )
        self.map_pub.publish(map_msg)


def main(args=None):
    """Entry point for the occupancy grid mapper node."""
    rclpy.init(args=args)
    node = OccupancyGridMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()