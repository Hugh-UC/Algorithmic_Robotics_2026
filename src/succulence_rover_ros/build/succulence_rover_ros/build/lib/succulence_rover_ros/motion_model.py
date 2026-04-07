"""
Probabilistic Motion Model for Odometry Processing (Week 5)

The /odom topic already provides the dead-reckoning pose — the robot's driver
integrates wheel encoder readings and publishes the accumulated X_t = X_{t-1} ⊕ u_t.
What /odom does NOT provide is the uncertainty (covariance) of that pose.

This module adds covariance tracking on top of the raw odometry. For each odom
step, it extracts the relative motion delta, computes how much noise that delta
introduces (the alpha noise model), and propagates the covariance via Jacobians.
The result: a dead-reckoning trajectory with a growing uncertainty ellipse that
demonstrates why SLAM is needed.

Student task:
  - Implement compute_motion_covariance() — the alpha noise model

References:
  - Thrun, Burgard, Fox, "Probabilistic Robotics" (2005), Chapter 5
  - Lecture 05, Section 5.3: Quantifying Odometry Uncertainty
"""

import numpy as np
from typing import List, Tuple, Optional
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, PoseStamped
from scipy.spatial.transform import Rotation

from . import utils


# ============================================================================
# STUDENT TODO: Implement this function
# ============================================================================

def compute_motion_covariance(relative_pose: np.ndarray,
                              alpha1: float, alpha2: float,
                              alpha3: float, alpha4: float) -> np.ndarray:
    """
    Compute covariance matrix for a relative motion using the alpha noise model.

    The key insight: odometry noise is proportional to how much the robot moved.
    A robot driving 10 metres accumulates more error than one driving 10 cm.
    A sharp turn is noisier than driving straight.

    Args:
        relative_pose: Relative motion [dx, dy, dtheta] in the robot's local frame
        alpha1: Translational noise from translation (wheel slip going straight)
        alpha2: Translational noise from rotation (positional drift when turning)
        alpha3: Rotational noise from translation (yaw drift going straight)
        alpha4: Rotational noise from rotation (encoder errors when turning)

    Returns:
        3x3 diagonal covariance matrix Q

    Algorithm:
        1. delta_trans = sqrt(dx^2 + dy^2)       # how far we translated
        2. delta_rot   = |dtheta|                 # how much we rotated
        3. var_x     = alpha1 * delta_trans^2 + alpha2 * delta_rot^2
           var_y     = alpha1 * delta_trans^2 + alpha2 * delta_rot^2
           var_theta = alpha3 * delta_trans^2 + alpha4 * delta_rot^2
        4. return np.diag([var_x, var_y, var_theta])

    Hints:
        - Use np.sqrt(), np.abs(), np.diag()
        - This function is ~8 lines of code
        - The result is used as Q in the covariance propagation formula:
          Sigma_new = J1 @ Sigma_old @ J1^T + J2 @ Q @ J2^T
    """
    # TODO: YOUR CODE HERE (~8 lines)
    dx, dy, dtheta = relative_pose

    # Compute motion magnitudes
    delta_trans = np.sqrt(dx**2 + dy**2)
    delta_rot = np.abs(dtheta)

    # Compute variances using alpha model
    var_x     = alpha1 * delta_trans**2 + alpha2 * delta_rot**2
    var_y     = alpha1 * delta_trans**2 + alpha2 * delta_rot**2
    var_theta = alpha3 * delta_trans**2 + alpha4 * delta_rot**2

    # Build diagonal covariance matrix
    cov = np.diag([var_x, var_y, var_theta])

    return cov


# ============================================================================
# Everything below is provided — you do not need to modify it.
# ============================================================================

class OdometryProcessor(Node):
    """
    Processes odometry messages and maintains a dead-reckoning trajectory
    with covariance propagation.

    Subscribes to raw odometry, chains SE(2) poses, and publishes:
      - /succulence/dead_reckoning/odometry  (Odometry with covariance ellipse)
      - /succulence/dead_reckoning/path      (Path trajectory for RViz)
    """

    def __init__(self):
        super().__init__('odometry_processor')

        # --- Parameters ---
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('dead_reckoning.odometry_topic',
                               '/succulence/dead_reckoning/odometry')
        self.declare_parameter('dead_reckoning.path_topic',
                               '/succulence/dead_reckoning/path')
        self.declare_parameter('frames.odom_frame', 'succulence/odom')
        self.declare_parameter('frames.base_link_frame', 'succulence/base_link')
        self.declare_parameter('frames.map_frame', 'map')
        self.declare_parameter('motion_model.alpha1', 0.01)
        self.declare_parameter('motion_model.alpha2', 0.005)
        self.declare_parameter('motion_model.alpha3', 0.005)
        self.declare_parameter('motion_model.alpha4', 0.02)
        self.declare_parameter('motion_model.max_trajectory_length', 5000)

        # --- Read parameters ---
        odom_topic = self.get_parameter('odom_topic').value
        odometry_out = self.get_parameter('dead_reckoning.odometry_topic').value
        path_out = self.get_parameter('dead_reckoning.path_topic').value

        self.odom_frame = self.get_parameter('frames.odom_frame').value
        self.base_link_frame = self.get_parameter('frames.base_link_frame').value
        self.map_frame = self.get_parameter('frames.map_frame').value

        self.alpha1 = self.get_parameter('motion_model.alpha1').value
        self.alpha2 = self.get_parameter('motion_model.alpha2').value
        self.alpha3 = self.get_parameter('motion_model.alpha3').value
        self.alpha4 = self.get_parameter('motion_model.alpha4').value
        self.max_trajectory_length = self.get_parameter(
            'motion_model.max_trajectory_length').value

        # --- State ---
        self.trajectory: List[Tuple[float, np.ndarray, np.ndarray]] = []
        self.prev_odom_pose: Optional[np.ndarray] = None
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_cov = np.zeros((3, 3))

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, odometry_out, 10)
        self.path_pub = self.create_publisher(Path, path_out, 10)

        # --- Subscriber ---
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)

        self.get_logger().info(f'OdometryProcessor started — listening on {odom_topic}')
        self.get_logger().info(
            f'Alpha params: a1={self.alpha1}, a2={self.alpha2}, '
            f'a3={self.alpha3}, a4={self.alpha4}')

    def odom_callback(self, msg: Odometry):
        """Process incoming odometry: update pose, propagate covariance, publish."""
        odom_pose = self._odom_msg_to_pose(msg)

        if self.prev_odom_pose is None:
            # First message — initialise
            self.prev_odom_pose = odom_pose
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.trajectory.append(
                (timestamp, self.current_pose.copy(), self.current_cov.copy()))
            return

        # Compute relative motion since last odometry message.
        # The /odom topic already provides the composed dead-reckoning pose —
        # we extract the delta here because we need it for two things:
        #   1. Computing Q (your compute_motion_covariance function)
        #   2. Computing the Jacobians for covariance propagation
        relative_odom = utils.pose_difference(self.prev_odom_pose, odom_pose)

        # Compute motion covariance using the alpha noise model
        # (This calls YOUR function from above!)
        motion_cov = compute_motion_covariance(
            relative_odom, self.alpha1, self.alpha2, self.alpha3, self.alpha4)

        if motion_cov is None:
            # Student hasn't implemented it yet — use zero covariance
            motion_cov = np.zeros((3, 3))

        # Propagate covariance: Σ_new = J₁ · Σ_old · J₁ᵀ + J₂ · Q · J₂ᵀ
        # This is the whole point of the motion model node — tracking how
        # uncertainty grows with each step. The pose itself comes from odom.
        J1, J2 = utils.pose_compose_jacobians(self.current_pose, relative_odom)
        self.current_cov = utils.covariance_propagate(
            self.current_cov, motion_cov, J1, J2)

        # Use the odom pose directly — the robot's driver already computes
        # X_t = X_{t-1} ⊕ u_t via wheel encoder integration.
        # (In Week 7, the SLAM node re-composes internally because graph
        # optimisation shifts poses away from raw odometry.)
        self.current_pose = odom_pose

        # Store in trajectory
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.trajectory.append(
            (timestamp, self.current_pose.copy(), self.current_cov.copy()))

        # Trim trajectory to prevent unbounded memory growth
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory = self.trajectory[-self.max_trajectory_length:]

        self.prev_odom_pose = odom_pose

        # Publish for RViz visualisation
        self._publish_odometry(msg.header.stamp)
        if len(self.trajectory) % 10 == 0:
            self._publish_path()

        # Periodic status log
        if len(self.trajectory) % 100 == 0:
            self.get_logger().info(
                f'Poses: {len(self.trajectory)}, '
                f'Position: [{self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}], '
                f'Cov trace: {np.trace(self.current_cov):.4f}')

    # --- Helper methods (provided) ---

    def _odom_msg_to_pose(self, msg: Odometry) -> np.ndarray:
        """Convert Odometry message to [x, y, theta]."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        theta = rotation.as_euler('xyz', degrees=False)[2]
        return np.array([x, y, theta])

    def _publish_odometry(self, stamp):
        """Publish dead-reckoning pose as Odometry (with covariance for RViz)."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_link_frame
        msg.pose.pose.position.x = self.current_pose[0]
        msg.pose.pose.position.y = self.current_pose[1]
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = self._yaw_to_quaternion(self.current_pose[2])
        msg.pose.covariance = self._3x3_to_6x6_covariance(self.current_cov)
        self.odom_pub.publish(msg)

    def _publish_path(self):
        """Publish dead-reckoning trajectory as Path for RViz."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame

        max_points = 500
        recent = self.trajectory[-max_points:]
        for _, pose, _ in recent:
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = pose[0]
            ps.pose.position.y = pose[1]
            ps.pose.position.z = 0.0
            ps.pose.orientation = self._yaw_to_quaternion(pose[2])
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        """Convert yaw angle to Quaternion message."""
        q = Rotation.from_euler('z', yaw).as_quat()
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def _3x3_to_6x6_covariance(self, cov_3x3: np.ndarray) -> list:
        """Convert 3x3 SE(2) covariance to 6x6 ROS covariance format."""
        cov = [0.0] * 36
        cov[0] = cov_3x3[0, 0]   # x-x
        cov[1] = cov_3x3[0, 1]   # x-y
        cov[6] = cov_3x3[1, 0]   # y-x
        cov[7] = cov_3x3[1, 1]   # y-y
        cov[35] = cov_3x3[2, 2]  # yaw-yaw
        return cov


def main(args=None):
    """Entry point for the motion model node."""
    rclpy.init(args=args)
    node = OdometryProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
