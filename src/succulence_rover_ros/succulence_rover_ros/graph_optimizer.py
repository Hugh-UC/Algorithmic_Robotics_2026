"""
Robust Levenberg-Marquardt Pose Graph Optimiser with Huber Kernels, with
Gauss-Newton Pose Graph Optimiser.
"""
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
# --- C++ Extensions ---
from scipy.optimize import least_squares
from scipy.linalg import sqrtm
# ----------------------

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


def jacobian_matrix(x, pose_graph, anchored_ids, num_edges, num_anchors):
    """
    Computes the sparse Jacobian analytically to pass to least_squares.
    """
    rows, cols, data = [], [], []
    n = len(x) // 3
    nodes = x.reshape((n, 3))
    
    # 1. Edge Jacobians
    for idx, (f_id, t_id, _, info_mat) in enumerate(pose_graph.edges):
        # Calculate your existing analytical jacobian blocks
        Ji, Jj = compute_jacobians(nodes[f_id], nodes[t_id])
        
        # Scaling by info_sqrt is required because residuals are scaled by info_sqrt
        info_sqrt = sqrtm(info_mat)
        Ji_weighted = info_sqrt @ Ji
        Jj_weighted = info_sqrt @ Jj
        
        row_offset = idx * 3
        col_i = f_id * 3
        col_j = t_id * 3
        
        for r in range(3):
            for c in range(3):
                rows.append(row_offset + r); cols.append(col_i + c); data.append(Ji_weighted[r, c])         # Ji block
                rows.append(row_offset + r); cols.append(col_j + c); data.append(Jj_weighted[r, c])         # Jj block

    # 2. Anchor Constraints (Using 3 degrees of freedom: X, Y, Theta)
    # Derivative of (Node - Anchor) * 1000.0 is 1000.0
    for idx, a_id in enumerate(anchored_ids):
        row_offset = 3 * num_edges + 3 * idx
        for r in range(3):
            rows.append(row_offset + r)
            cols.append(a_id * 3 + r)
            data.append(1000.0)

    total_residuals = 3 * num_edges + 3 * num_anchors
    return sparse.coo_matrix((data, (rows, cols)), shape=(total_residuals, 3 * n)).tocsr()


def norm_angle_vec(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


def vectorize_original(x, num_nodes, from_ids, to_ids, meas, sqrt_omegas, anchored_ids, anchor_poses, num_anchors):
        nodes = x.reshape((num_nodes, 3))
        
        # --- Edge Residuals ---
        pi = nodes[from_ids]
        pj = nodes[to_ids]
        
        c = np.cos(pi[:, 2])
        s = np.sin(pi[:, 2])
        
        dx = pj[:, 0] - pi[:, 0]
        dy = pj[:, 1] - pi[:, 1]
        
        diff_x = c * dx + s * dy
        diff_y = -s * dx + c * dy
        diff_theta = norm_angle_vec(pj[:, 2] - pi[:, 2])
        
        err = np.column_stack((diff_x, diff_y, diff_theta)) - meas
        err[:, 2] = norm_angle_vec(err[:, 2])
        
        # Apply Information Matrix weighting instantly via C-einsum
        err_weighted = np.einsum('nij,nj->ni', sqrt_omegas, err).flatten()
        
        # --- Anchor Residuals ---
        if num_anchors > 0:
            a_diff = nodes[anchored_ids] - anchor_poses
            a_diff[:, 2] = norm_angle_vec(a_diff[:, 2])
            a_err = (a_diff * 1000.0).flatten() # 1000^2 = 1e6 Anchor weight
            return np.concatenate((err_weighted, a_err))
            
        return err_weighted


def optimize_lm(pose_graph: PoseGraph, num_iterations: int = 10, window_size: int = 0):
    """
    Optimises the pose graph using a robust M-estimator and Levenberg-Marquardt damping.
    Fully tolerates false loop closures and mitigates gauge freedom.

    Args:
        pose_graph (PoseGraph): _description_
        num_iterations (int, optional): _description_. Defaults to 10.
    """
    n = pose_graph.get_num_nodes()
    
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


        if window_size > 0 and n > window_size:
            # Anchor ALL nodes outside the active window (Infinite Mass)
            start_idx = n - window_size
            for i in range(start_idx):
                H[i*3:i*3+3, i*3:i*3+3] += sparse.eye(3) * 1e6
        else:
            # Global Optimization: Anchor first node only
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


def optimize_c(pose_graph: PoseGraph, num_iterations: int = 10, window_size: int = 0):
    """
    C-Engine backed optimizer using SciPy's least_squares (TRF/MINPACK).
    Uses Vectorized Numpy arrays to eliminate Python loops.
    """
    n = pose_graph.get_num_nodes()

    num_edges = len(pose_graph.edges)
    huber_k = 2.0

    # 1. Pre-allocate Vectorized Arrays
    from_ids = np.zeros(num_edges, dtype=int)
    to_ids = np.zeros(num_edges, dtype=int)
    meas = np.zeros((num_edges, 3))
    sqrt_omegas = np.zeros((num_edges, 3, 3))

    for idx, (f_id, t_id, m, omega) in enumerate(pose_graph.edges):
        from_ids[idx] = f_id
        to_ids[idx] = t_id
        meas[idx] = m
        # Calculate matrix square root of information matrix once
        sqrt_omegas[idx] = np.real(sqrtm(omega))

    # 2. Setup Anchors
    anchored_ids = []
    anchor_poses = []
    if window_size > 0 and n > window_size:
        start_idx = n - window_size
        for i in range(start_idx):
            anchored_ids.append(i)
            anchor_poses.append(pose_graph.nodes[i].copy())
    else:
        anchored_ids.append(0)
        anchor_poses.append(pose_graph.nodes[0].copy())

    anchored_ids = np.array(anchored_ids, dtype=int)
    anchor_poses = np.array(anchor_poses)
    num_anchors = len(anchored_ids)

    # 3. Vectorized Residual Function (NO PYTHON LOOPS!)
    def vectorize_func(x):
        return vectorize_original(x, n, from_ids, to_ids, meas, sqrt_omegas, anchored_ids, anchor_poses, num_anchors)

    # 4. Initial Guess
    x0 = np.array([p for p in pose_graph.nodes]).flatten()

    # 5. Define Analytical Jacobian Callback
    def jac_func(x):
        return jacobian_matrix(x, pose_graph, anchored_ids, num_edges, num_anchors)

    # 6. Execute C-Engine Optimization
    res = least_squares(
        vectorize_func, x0,
        jac=jac_func,   # type: ignore
        loss='huber', f_scale=huber_k,
        method='trf',
        max_nfev=max(20, num_iterations * 3),
        ftol=1e-4, xtol=1e-4
    )

    # 7. Apply Results back to Graph
    optimized_nodes = res.x.reshape((n, 3))
    for i in range(n):
        pose_graph.nodes[i][0] = optimized_nodes[i, 0]
        pose_graph.nodes[i][1] = optimized_nodes[i, 1]
        pose_graph.nodes[i][2] = norm_angle_vec(optimized_nodes[i, 2])


def optimize(pose_graph: PoseGraph, num_iterations: int = 10, window_size: int = 0, use_c_engine: bool = False):
    """
    Master Router Function
    Args:
        pose_graph (PoseGraph): _description_
        num_iterations (int, optional): _description_. Defaults to 10.
        use_fast_math (bool, optional): _description_. Defaults to False.
    """
    if pose_graph.get_num_nodes() < 2 or pose_graph.get_num_edges() == 0:
        return

    if use_c_engine:
        optimize_c(pose_graph, num_iterations, window_size)
    else:
        optimize_lm(pose_graph, num_iterations, window_size)