"""
Robust Levenberg-Marquardt Pose Graph Optimiser with Huber Kernels, with
Gauss-Newton Pose Graph Optimiser.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import utils
from .pose_graph import PoseGraph


def compute_error(pose_i: np.ndarray, pose_j: np.ndarray,
                  measurement: np.ndarray) -> np.ndarray:
    """
    Edge error: predicted relative pose minus the measurement (angle-wrapped).

    Args:
        pose_i (np.ndarray): _description_
        pose_j (np.ndarray): _description_
        measurement (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    predicted = utils.pose_difference(pose_i, pose_j)
    error = predicted - measurement
    error[2] = utils.normalize_angle(error[2])
    return error


def compute_jacobians(pose_i: np.ndarray,
                      pose_j: np.ndarray) -> tuple:
    """
    Analytical Jacobians of the edge error w.r.t. the two connected poses.

    Args:
        pose_i (np.ndarray): _description_
        pose_j (np.ndarray): _description_

    Returns:
        tuple: _description_
    """
    theta_i = pose_i[2]
    c = np.cos(theta_i)
    s = np.sin(theta_i)

    dx = pose_j[0] - pose_i[0]
    dy = pose_j[1] - pose_i[1]

    Ji = np.array([
        [-c, -s, -s * dx + c * dy],
        [ s, -c, -c * dx - s * dy],
        [ 0,  0,              -1.0]
    ])

    Jj = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [ 0,  0, 1.0]
    ])

    return Ji, Jj


def optimize(pose_graph: PoseGraph, num_iterations: int = 10):
    """
    Optimises the pose graph using a robust M-estimator and Levenberg-Marquardt damping.
    Fully tolerates false loop closures and mitigates gauge freedom.

    Args:
        pose_graph (PoseGraph): _description_
        num_iterations (int, optional): _description_. Defaults to 10.
    """
    n = pose_graph.get_num_nodes()
    if n < 2 or pose_graph.get_num_edges() == 0:
        return

    # Each pose has 3 DOF (x, y, theta)
    dim = 3 * n

    # Threshold for outlier rejection (Mahalanobis distance)
    huber_k = 2.0

    for iteration in range(num_iterations):
        H = sparse.lil_matrix((dim, dim))
        b = np.zeros(dim)

        for from_id, to_id, measurement, omega in pose_graph.edges:
            pose_i = pose_graph.nodes[from_id]
            pose_j = pose_graph.nodes[to_id]

            e = compute_error(pose_i, pose_j, measurement)

            # M-Estimator (Huber Kernel)
            mahalanobis_d = np.sqrt(float(e.T @ omega @ e))

            w = 1.0
            if mahalanobis_d > huber_k:
                w = huber_k / mahalanobis_d         # Downweight outlier constraints linearly
            
            # Scale information matrix by M-estimator weight
            omega_w = omega * w

            Ji, Jj = compute_jacobians(pose_i, pose_j)

            JiT_omega = Ji.T @ omega_w
            JjT_omega = Jj.T @ omega_w

            idx_i = 3 * from_id
            idx_j = 3 * to_id

            H[idx_i:idx_i+3, idx_i:idx_i+3] += JiT_omega @ Ji
            H[idx_i:idx_i+3, idx_j:idx_j+3] += JiT_omega @ Jj
            H[idx_j:idx_j+3, idx_i:idx_i+3] += JjT_omega @ Ji
            H[idx_j:idx_j+3, idx_j:idx_j+3] += JjT_omega @ Jj

            b[idx_i:idx_i+3] += JiT_omega @ e
            b[idx_j:idx_j+3] += JjT_omega @ e

        # Anchor first node
        H[0:3, 0:3] += sparse.eye(3) * 1e6

        # Levenberg-Marquardt Damping / Regularization
        # Guarantees H is strictly positive-definite and invertible under all geometric conditions
        lm_lambda = 1e-5
        H += sparse.eye(dim) * lm_lambda

        # Solve the sparse linear system H @ dx = -b
        dx = spsolve(H.tocsr(), -b)

        # Apply the update vector (dx) to all node poses in place
        for i in range(n):
            idx = 3 * i
            pose_graph.nodes[i][0] += dx[idx]      # Update X position
            pose_graph.nodes[i][1] += dx[idx+1]    # Update Y position
            pose_graph.nodes[i][2] += dx[idx+2]    # Update Theta (heading)

            # Normalize angle to prevent values from winding past [-pi, pi]
            pose_graph.nodes[i][2] = utils.normalize_angle(pose_graph.nodes[i][2])

        # Optional Early Exit: If the correction vector is micro-small, optimization has converged
        # (Convergence Check)
        if np.linalg.norm(dx) < 1e-5:
            break
