"""
Gauss-Newton Pose Graph Optimiser (Week 11)

PROVIDED COMPLETE — DO NOT MODIFY.

Gauss-Newton optimisation will be covered in depth in Week 12. For this lab,
treat this file as a working black box: the SLAM node calls `optimize(graph,
n_iters)` and the graph nodes are updated in place. Read through it if you are
curious about the maths — it builds a sparse linear system H @ dx = -b from
the edge constraints, solves it, applies the update, and repeats.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import utils
from .pose_graph import PoseGraph


def compute_error(pose_i: np.ndarray, pose_j: np.ndarray,
                  measurement: np.ndarray) -> np.ndarray:
    """Edge error: predicted relative pose minus the measurement (angle-wrapped)."""
    predicted = utils.pose_difference(pose_i, pose_j)
    error = predicted - measurement
    error[2] = utils.normalize_angle(error[2])
    return error


def compute_jacobians(pose_i: np.ndarray,
                      pose_j: np.ndarray) -> tuple:
    """Analytical Jacobians of the edge error w.r.t. the two connected poses."""
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
    Optimise pose graph using Gauss-Newton least-squares.
    Modifies pose_graph in place.
    """
    n = pose_graph.get_num_nodes()
    if n < 2 or pose_graph.get_num_edges() == 0:
        return

    dim = 3 * n

    for iteration in range(num_iterations):
        H = sparse.lil_matrix((dim, dim))
        b = np.zeros(dim)

        for from_id, to_id, measurement, omega in pose_graph.edges:
            pose_i = pose_graph.nodes[from_id]
            pose_j = pose_graph.nodes[to_id]

            e = compute_error(pose_i, pose_j, measurement)
            Ji, Jj = compute_jacobians(pose_i, pose_j)

            JiT_omega = Ji.T @ omega
            JjT_omega = Jj.T @ omega

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

        H_csc = H.tocsc()
        try:
            dx = spsolve(H_csc, -b)
        except Exception:
            break

        for i in range(n):
            idx = 3 * i
            pose_graph.nodes[i][0] += dx[idx]
            pose_graph.nodes[i][1] += dx[idx + 1]
            pose_graph.nodes[i][2] = utils.normalize_angle(
                pose_graph.nodes[i][2] + dx[idx + 2])

        if np.linalg.norm(dx) < 1e-6:
            break
