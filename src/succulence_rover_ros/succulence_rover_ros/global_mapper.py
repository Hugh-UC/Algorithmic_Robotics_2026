"""
Global Mapper Node (The Artist)

Subscribes to raw laser scans and the optimized Pose Graph Path.
Constructs and publishes the Occupancy Grid without blocking the Estimator node.

Usage: This node runs concurrently with the slam_estimator node to build
       and maintain a ghost-wall-free occupancy grid map from optimized keyframe poses.
"""
import numpy as np
import time
import threading
import copy
import rclpy
from rclpy.node import Node, Optional
from nav_msgs.msg import Path, OccupancyGrid as OccupancyGridMsg
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int32
from scipy.spatial.transform import Rotation

from .occupancy_grid_mapper import OccupancyGrid

class GlobalMapperNode(Node):
    def __init__(self):
        super().__init__('global_mapper')

        param_defaults: dict[str, str | int | float | bool] = {
            'scan_topic': '/scan',                      # Physical raw Lidar
            'odom_topic': '/odom',                      # Physical raw Odometry
            'map_topic': '/succulence/map',
            'slam_odometry_topic': '/succulence/slam/odometry',
            'slam_path_topic': '/succulence/slam/path',
            'keyframe_scan_topic': '/succulence/slam/keyframe_scan',
            'map_version_topic': '/succulence/slam/version',
            
            'slam.keyframe_window': 100,                # Keyframe rendering window for map updates
            'slam.map_publish_interval': 0.3,           # Fast refresh for physical safety
            'slam.scan_buffer_max_size': 5000,          # Larger: more history for optimization, but more RAM usage.

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


        self.scan_topic             = str(get_p('scan_topic'))
        self.slam_path_topic        = str(get_p('slam_path_topic'))
        self.map_topic              = str(get_p('map_topic'))
        map_publish_interval        = float(get_p('slam.map_publish_interval'))
        self.keyframe_scan_topic    = str(get_p('keyframe_scan_topic'))         # Define topic for filtered keyframe scans
        self.map_version_topic      = str(get_p('map_version_topic'))           # Define topic for map versioning

        self.keyframe_window        = int(get_p('slam.keyframe_window'))

        # Memory buffer for historical laser scans
        # Key: timestamp (seconds as float), Value: LaserScan message
        self.scan_buffer = {}
        self.scan_buffer_max_size = int(get_p('slam.scan_buffer_max_size'))

        # Init Grid Map (Use your params here)
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

        self.last_path_len = 0
        self.cached_poses = []

        self.target_map_version = 0
        self.published_map_version = 0
        self.latest_path_msg = None

        self.map_ready = False

        # Publishers / Subscribers
        #self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 100)
        self.create_subscription(LaserScan, self.keyframe_scan_topic, self._scan_cb, 10)
        self.create_subscription(Path, self.slam_path_topic, self._path_cb, 10)

        self.map_pub = self.create_publisher(OccupancyGridMsg, self.map_topic, 10)

        self.map_timer = self.create_timer(map_publish_interval, self._publish_map)

        self.version_sub = self.create_subscription(Int32, self.map_version_topic, self._version_cb, 10)
        self.map_ack_pub = self.create_publisher(Int32, '/succulence/map_ack', 10)

        self.get_logger().info("Global Mapper (Artist) Started.")

        self.rebuild_lock = threading.Lock()


    def _version_cb(self, msg: Int32):
        """
        Updates the internal target version when the Estimator finishes optimizing.
        """
        self.target_map_version = msg.data
        self.get_logger().info(f"Master optimisation v{self.target_map_version} received. Triggering map rebuild.")

        if self.latest_path_msg is not None:
             threading.Thread(target=self._rebuild_map, args=(self.latest_path_msg,), daemon=True).start()


    def _scan_cb(self, msg: LaserScan):
        """
        Stores historical laser scans, keyed by their exact timestamp in nanoseconds.
        """
        # Convert to integer nanoseconds (consistent with ROS 2 native format)
        t_scan = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec

        self.scan_buffer[t_scan] = msg
        
        # Garbage collection: keep the last 'N' scans to prevent RAM bloat
        while len(self.scan_buffer) > self.scan_buffer_max_size:
            oldest_t = min(self.scan_buffer.keys())
            del self.scan_buffer[oldest_t]


    def _path_cb(self, msg: Path):
        """
        Handles incoming path updates. Triggers an incremental draw for new keyframes,
        or a full background rebuild if the graph was optimized (shifted).
        """
        self.latest_path_msg = msg

        if not msg.poses:
            return

        current_len = len(msg.poses)

        # 1. No change
        if current_len == self.last_path_len:
            return

        # 2. Incremental Addition (single new keyframe was added, no optimization yet)
        if current_len == self.last_path_len + 1:
            latest_pose_msg = msg.poses[-1]
            stamp_nanos = latest_pose_msg.header.stamp.sec * 1000000000 + latest_pose_msg.header.stamp.nanosec

            # Fuzzy matcher
            scan_msg = self._get_scan_at_time(stamp_nanos)
            
            if scan_msg:
                x = latest_pose_msg.pose.position.x
                y = latest_pose_msg.pose.position.y
                yaw = self._quaternion_to_yaw(latest_pose_msg.pose.orientation)
                
                self.occupancy_grid.update(
                    pose=np.array([x, y, yaw]),
                    ranges=np.array(scan_msg.ranges),
                    angle_min=scan_msg.angle_min,
                    angle_increment=scan_msg.angle_increment
                )

        self.last_path_len = current_len
        self.cached_poses = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]


    def _get_scan_at_time(self, target_nanos: int, tolerance_ns: int = 10000000) -> Optional[LaserScan]:
        """
        Gets a laser scan at a specific timestamp, with a tolerance.

        Args:
            target_nanos (int): The target timestamp in nanoseconds.
            tolerance_ns (int, optional): The tolerance in nanoseconds. Defaults to 10000000.

        Returns:
            Optional[LaserScan]: _description_
        """
        # Look for a scan within 10ms (10,000,000 ns)
        for t, scan in self.scan_buffer.items():
            if abs(t - target_nanos) < tolerance_ns:
                return scan
        return None


    def _rebuild_map(self, path_msg: Path):
        t0 = time.monotonic()

        # Capture version thread is currently drawing
        version_being_built = self.target_map_version

        # 1. Determine which keyframes to render based on the configured window size.
        total_poses : int   = len(path_msg.poses)
        window : int        = self.keyframe_window

        if window > 0 and total_poses > window:
            poses_to_render = path_msg.poses[-window:]
        else:
            poses_to_render = path_msg.poses
        
        # 2. Create an independent 'off-screen' mapper for the background thread
        bg_mapper = copy.deepcopy(self.occupancy_grid)
        bg_mapper.grid.fill(0.0)
        
        # 3. Render all keyframes into the background grid using the Path message
        n_rendered = 0
        for pose_stamped in poses_to_render:
            stamp_nanos = pose_stamped.header.stamp.sec * 1000000000 + pose_stamped.header.stamp.nanosec

            if stamp_nanos not in self.scan_buffer:
                self.get_logger().warn(f"Skipping keyframe pose; scan {stamp_nanos} not found in buffer.")
                continue
            
            if stamp_nanos in self.scan_buffer:
                scan_msg = self.scan_buffer[stamp_nanos]
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                yaw = self._quaternion_to_yaw(pose_stamped.pose.orientation)
                
                bg_mapper.update(
                    pose=np.array([x, y, yaw]), 
                    ranges=np.array(scan_msg.ranges),
                    angle_min=scan_msg.angle_min, 
                    angle_increment=scan_msg.angle_increment
                )
                n_rendered += 1
            
        # 4. Atomically swap the fully built grid back to the live system!
        with self.rebuild_lock:
            self.occupancy_grid.grid = bg_mapper.grid

            self.published_map_version = version_being_built
            self.map_ready = True

        n_known = int(np.sum(np.abs(self.occupancy_grid.grid) > 0.1))
        self.get_logger().info(
            f' 🏁 Background Map Rebuild Complete: {n_rendered} keyframes rendered, '
            f'{n_known} known cells, {time.monotonic() - t0:.2f}s')
        
    
    def _publish_map(self):
        if self.last_path_len == 0:
            return
        
        t0 = time.monotonic()

        with self.rebuild_lock:
            map_msg = self.occupancy_grid.to_ros_message(frame_id='map', timestamp=self.get_clock().now().to_msg())

            self.map_pub.publish(map_msg)

        # Announce to Estimator that map version is ready
        self.map_ack_pub.publish(Int32(data=self.published_map_version))

        n_known = int(np.sum(np.abs(self.occupancy_grid.grid) > 0.1))

        self.get_logger().info(
            f' 🗺️ Map published: {n_known} known cells '
            f'({100.0*n_known/self.occupancy_grid.grid.size:.2f}% of grid), '
            f'{time.monotonic() - t0:.2f}s')
        
    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]


def main(args=None):
    rclpy.init(args=args)
    node = GlobalMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Shutting down Global Mapper cleanly. '
            f'Final path length: {node.last_path_len} keyframes, '
            f'Scans currently in buffer: {len(node.scan_buffer)}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()