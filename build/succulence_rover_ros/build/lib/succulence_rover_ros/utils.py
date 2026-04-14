"""
SE(2) Utilities for 2D Pose Mathematics

All poses are represented as NumPy arrays [x, y, theta].

Key operations:
  - pose_compose(p1, p2):   Chain two transformations (p1 then p2)
  - pose_inverse(p):        Invert a transformation
  - pose_difference(p1, p2): Relative pose from p1 to p2
  - pose_compose_jacobians:  Derivatives for covariance propagation
  - covariance_propagate:    Propagate uncertainty through composition

These functions are the mathematical building blocks for the entire SLAM system.
You used them in the Week 4 Jupyter lab — here they are in their ROS-ready form.

References:
  - Sola, "A micro Lie theory for state estimation in robotics" (2021)
  - Lecture 05, Section 5.2: SE(2) Robot Pose Mathematics
"""

import numpy as np
from typing import Tuple


def normalize_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while theta > np.pi:
        theta -= 2.0 * np.pi
    while theta < -np.pi:
        theta += 2.0 * np.pi
    return theta


def rotation_matrix(theta: float) -> np.ndarray:
    """Create 2x2 rotation matrix from angle."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def pose_compose(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compose two SE(2) poses: result = p1 (compose) p2.

    If you are at pose p1 and move by p2 (in p1's local frame),
    this computes your new pose in the global frame.

    Equivalent to T(result) = T(p1) @ T(p2) in matrix form.
    """
    x1, y1, theta1 = p1
    x2, y2, theta2 = p2

    R1 = rotation_matrix(theta1)
    t2_rotated = R1 @ np.array([x2, y2])

    x = x1 + t2_rotated[0]
    y = y1 + t2_rotated[1]
    theta = normalize_angle(theta1 + theta2)

    return np.array([x, y, theta])


def pose_inverse(p: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of an SE(2) pose.

    If p transforms from frame A to frame B,
    pose_inverse(p) transforms from B to A.

    Verification: pose_compose(p, pose_inverse(p)) == [0, 0, 0]
    """
    x, y, theta = p

    theta_inv = -theta
    R_inv = rotation_matrix(theta_inv)
    t_inv = -R_inv @ np.array([x, y])

    return np.array([t_inv[0], t_inv[1], normalize_angle(theta_inv)])


def pose_difference(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from p1 to p2.

    Equivalent to: pose_compose(pose_inverse(p1), p2)
    This is the (ominus) operator: result = p1 (ominus) p2.
    """
    return pose_compose(pose_inverse(p1), p2)


def pose_compose_jacobians(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobians of pose composition p1 (compose) p2.

    Returns:
        J1: 3x3 Jacobian d(p1+p2)/dp1
        J2: 3x3 Jacobian d(p1+p2)/dp2

    Used for covariance propagation through composition.
    """
    x1, y1, theta1 = p1
    x2, y2, theta2 = p2

    c1 = np.cos(theta1)
    s1 = np.sin(theta1)

    J1 = np.array([
        [1.0, 0.0, -s1 * x2 - c1 * y2],
        [0.0, 1.0,  c1 * x2 - s1 * y2],
        [0.0, 0.0,  1.0]
    ])

    J2 = np.array([
        [ c1, -s1, 0.0],
        [ s1,  c1, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return J1, J2


def covariance_propagate(cov_p1: np.ndarray, cov_p2: np.ndarray,
                        J1: np.ndarray, J2: np.ndarray) -> np.ndarray:
    """
    Propagate covariance through pose composition.

    Formula: Sigma_result = J1 @ Sigma1 @ J1^T + J2 @ Sigma2 @ J2^T
    """
    return J1 @ cov_p1 @ J1.T + J2 @ cov_p2 @ J2.T
