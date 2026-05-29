"""
Robust Levenberg-Marquardt Pose Graph Optimiser with Huber Kernels, with
Gauss-Newton Pose Graph Optimiser.
"""
import numpy as np
from typing import Any
from scipy import sparse
from scipy.sparse.linalg import spsolve
# --- C++ Extensions (full) ---
from scipy.optimize import least_squares
from scipy.linalg import sqrtm
# ----------------------

from . import utils

# ---- C++ Extension (cpp) ----
# Attempt to load the compiled
# C++ extension
try:
    import succulence_cpp_optimizer
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False
    print("\n[WARNING] C++ Optimizer not found. Falling back to Python loop.\n")
# ---------------------------


def compute_error(pose_i: np.ndarray, pose_j: np.ndarray, measurement: np.ndarray) -> np.ndarray:
    """
    Edge error: predicted relative pose minus the measurement (angle-wrapped).

    Args:
        pose_i (np.ndarray): Origin pose of the edge [x, y, theta].
        pose_j (np.ndarray): Target pose of the edge [x, y, theta].
        measurement (np.ndarray): The expected measurement constraint [x, y, theta].

    Returns:
        np.ndarray: The resulting error residual.
    """
    predicted = utils.pose_difference(pose_i, pose_j)
    error = predicted - measurement
    error[2] = utils.normalize_angle(error[2])
    return error


def compute_jacobians(pose_i: np.ndarray, pose_j: np.ndarray) -> tuple:
    """
    Analytical Jacobians of the edge error w.r.t. the two connected poses.

    Args:
        pose_i (np.ndarray): Origin pose [x, y, theta].
        pose_j (np.ndarray): Target pose [x, y, theta].

    Returns:
        tuple: A tuple containing (Jacobian w.r.t pose_i, Jacobian w.r.t pose_j).
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


def jacobian_matrix(x : np.ndarray, num_nodes : int, from_ids : np.ndarray, to_ids : np.ndarray, sqrt_omegas : np.ndarray, jac_rows : np.ndarray, jac_cols : np.ndarray, num_edges : int, num_anchors : int):
    """
    Computes the sparse Jacobian analytically to pass to SciPy's least_squares.

    Args:
        x (np.ndarray): Flat array of all node states.
        num_nodes (int): Total number of nodes.
        from_ids (np.ndarray): Array of origin node IDs for edges.
        to_ids (np.ndarray): Array of target node IDs for edges.
        sqrt_omegas (np.ndarray): Array of square-rooted information matrices.
        jac_rows (np.ndarray): Pre-computed row indices for sparse matrix.
        jac_cols (np.ndarray): Pre-computed column indices for sparse matrix.
        num_edges (int): Total number of edges.
        num_anchors (int): Total number of anchor constraints.

    Returns:
        scipy.sparse.csr_matrix: The sparse Jacobian matrix.
    """
    nodes = x.reshape((num_nodes, 3))
    pi = nodes[from_ids]
    pj = nodes[to_ids]
    
    c = np.cos(pi[:, 2])
    s = np.sin(pi[:, 2])
    dx = pj[:, 0] - pi[:, 0]
    dy = pj[:, 1] - pi[:, 1]
    
    # Batch construct analytical derivative matrix blocks for all edges simultaneously
    Ji = np.zeros((num_edges, 3, 3))
    Ji[:, 0, 0] = -c
    Ji[:, 0, 1] = -s
    Ji[:, 0, 2] = -s * dx + c * dy
    Ji[:, 1, 0] = s
    Ji[:, 1, 1] = -c
    Ji[:, 1, 2] = -c * dx - s * dy
    Ji[:, 2, 2] = -1.0
    
    Jj = np.zeros((num_edges, 3, 3))
    Jj[:, 0, 0] = c
    Jj[:, 0, 1] = s
    Jj[:, 1, 0] = -s
    Jj[:, 1, 1] = c
    Jj[:, 2, 2] = 1.0
    
    # Batch scale all derivatives via single NumPy C-backed matrix multiplication
    Ji_weighted = np.matmul(sqrt_omegas, Ji)
    Jj_weighted = np.matmul(sqrt_omegas, Jj)
    
    # Reshape data array to perfectly slide into our precomputed layout structure
    Ji_flat = Ji_weighted.reshape(num_edges, 9)
    Jj_flat = Jj_weighted.reshape(num_edges, 9)
    edges_flat = np.hstack((Ji_flat, Jj_flat)).flatten()
    
    # Constant anchor derivative block
    anchor_flat = np.full(3 * num_anchors, 1000.0)
    
    # Combine data arrays
    jac_data = np.concatenate((edges_flat, anchor_flat))
    
    # Return exact CSR compressed matrix format instantly
    return sparse.coo_matrix(
        (jac_data, (jac_rows, jac_cols)), 
        shape=(3 * num_edges + 3 * num_anchors, 3 * num_nodes)
    ).tocsr()


def norm_angle_vec(th : np.ndarray) -> np.ndarray:
    """
    Normalizes an array of angles to the [-pi, pi] range.

    Args:
        th (np.ndarray): Array of angles in radians.

    Returns:
        np.ndarray: Array of normalized angles.
    """
    return (th + np.pi) % (2 * np.pi) - np.pi


def vectorize_original(x : np.ndarray, num_nodes : int, from_ids : np.ndarray, to_ids : np.ndarray, meas : np.ndarray, sqrt_omegas : np.ndarray, anchored_ids : np.ndarray, anchor_poses : np.ndarray, num_anchors : int):
    """
    Vectorized residual function for the C-engine least_squares optimizer.

    Args:
        x (np.ndarray): Flat array of all node states.
        num_nodes (int): Total number of nodes in the graph.
        from_ids (np.ndarray): Array of origin node IDs.
        to_ids (np.ndarray): Array of target node IDs.
        meas (np.ndarray): Array of measurement vectors.
        sqrt_omegas (np.ndarray): Array of square-rooted information matrices.
        anchored_ids (np.ndarray): Array of IDs for anchored nodes.
        anchor_poses (np.ndarray): Array of fixed poses for anchored nodes.
        num_anchors (int): Total number of anchor constraints.

    Returns:
        np.ndarray: A concatenated array of edge and anchor residual errors.
    """
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


def optimize_lm(snapshot : dict[str, Any], num_iterations : int = 10, window_size : int = 0):
    """
    Optimises the pose graph using a robust M-estimator and Levenberg-Marquardt damping.
    Fully tolerates false loop closures and mitigates gauge freedom.

    Args:
        snapshot (dict[str, Any]): Snapshot dictionary containing 'nodes' and 'edges'.
        num_iterations (int, optional): Optimization iterations. Defaults to 10.
        window_size (int, optional): Size of the active sliding window. Defaults to 0.
    """
    n : int         = len(snapshot['nodes'])
    dim : int       = 3 * n         # Each pose has 3 DOF (x, y, theta)
    huber_k : float = 2.0           # Threshold for outlier rejection (Mahalanobis distance)

    for iteration in range(num_iterations):
        H = sparse.lil_matrix((dim, dim))
        b = np.zeros(dim)

        for from_id, to_id, measurement, omega in snapshot['edges']:
            pose_i = snapshot['nodes'][from_id]
            pose_j = snapshot['nodes'][to_id]

            e = compute_error(pose_i, pose_j, measurement)

            # M-Estimator (Huber Kernel)
            mahalanobis_d = np.sqrt(float(e.T @ omega @ e))

            w : float = 1.0
            if mahalanobis_d > huber_k:
                w = huber_k / mahalanobis_d         # Downweight outlier constraints linearly
            
            # Scale information matrix by M-estimator weight
            omega_w : np.ndarray = omega * w

            Ji, Jj = compute_jacobians(pose_i, pose_j)

            JiT_omega : np.ndarray = Ji.T @ omega_w
            JjT_omega : np.ndarray = Jj.T @ omega_w

            idx_i : int = 3 * from_id
            idx_j : int = 3 * to_id

            H[idx_i:idx_i+3, idx_i:idx_i+3] += JiT_omega @ Ji
            H[idx_i:idx_i+3, idx_j:idx_j+3] += JiT_omega @ Jj
            H[idx_j:idx_j+3, idx_i:idx_i+3] += JjT_omega @ Ji
            H[idx_j:idx_j+3, idx_j:idx_j+3] += JjT_omega @ Jj

            b[idx_i:idx_i+3] += JiT_omega @ e
            b[idx_j:idx_j+3] += JjT_omega @ e


        if window_size > 0 and n > window_size:
            # Anchor ALL nodes outside the active window (Infinite Mass)
            start_idx : int = n - window_size
            for i in range(start_idx):
                H[i*3:i*3+3, i*3:i*3+3] += sparse.eye(3) * 1e6
        else:
            # Global Optimization: Anchor first node only
            H[0:3, 0:3] += sparse.eye(3) * 1e6

        # Levenberg-Marquardt Damping / Regularization
        # Guarantees H is strictly positive-definite and invertible under all geometric conditions
        lm_lambda : float = 1e-5
        H += sparse.eye(dim) * lm_lambda

        # Solve the sparse linear system H @ dx = -b
        dx = spsolve(H.tocsr(), -b)

        # Apply the update vector (dx) to all node poses in place
        for i in range(n):
            idx : int = 3 * i
            snapshot['nodes'][i][0] += dx[idx]      # Update X position
            snapshot['nodes'][i][1] += dx[idx+1]    # Update Y position
            snapshot['nodes'][i][2] += dx[idx+2]    # Update Theta (heading)

            # Normalize angle to prevent values from winding past [-pi, pi]
            snapshot['nodes'][i][2] = utils.normalize_angle(snapshot['nodes'][i][2])

        # Optional Early Exit: If the correction vector is micro-small, optimization has converged
        # (Convergence Check)
        if np.linalg.norm(dx) < 1e-5:
            break


def optimize_c(snapshot : dict[str, Any], num_iterations : int = 10, window_size : int = 0):
    """
    C-Engine backed optimizer using SciPy's least_squares (TRF/MINPACK).
    Uses Vectorized Numpy arrays to eliminate Python loops.

    Args:
        snapshot (dict[str, Any]): Snapshot dictionary containing 'nodes' and 'edges'.
        num_iterations (int, optional): Optimization iterations. Defaults to 10.
        window_size (int, optional): Size of the active sliding window. Defaults to 0.
    """
    n : int         = len(snapshot['nodes'])
    num_edges : int = len(snapshot['edges'])
    huber_k : float = 2.0

    # 1. Pre-allocate Vectorized Arrays
    from_ids : np.ndarray       = np.zeros(num_edges, dtype=int)
    to_ids : np.ndarray         = np.zeros(num_edges, dtype=int)
    meas : np.ndarray           = np.zeros((num_edges, 3))
    sqrt_omegas : np.ndarray    = np.zeros((num_edges, 3, 3))

    for idx, (f_id, t_id, m, omega) in enumerate(snapshot['edges']):
        from_ids[idx] = f_id
        to_ids[idx] = t_id
        meas[idx] = m
        # Calculate matrix square root of information matrix once
        sqrt_omegas[idx] = np.real(sqrtm(omega))

    # 2. Setup Anchors
    anchored_ids = []
    anchor_poses = []
    if window_size > 0 and n > window_size:
        start_idx : int = n - window_size
        for i in range(start_idx):
            anchored_ids.append(i)
            anchor_poses.append(snapshot['nodes'][i].copy())
    else:
        anchored_ids.append(0)
        anchor_poses.append(snapshot['nodes'][0].copy())

    anchored_ids = np.array(anchored_ids, dtype=int)
    anchor_poses = np.array(anchor_poses)
    num_anchors : int = len(anchored_ids)

    # 3. Vectorized Residual Function (NO PYTHON LOOPS!)
    def vectorize_func(x):
        return vectorize_original(x, n, from_ids, to_ids, meas, sqrt_omegas, anchored_ids, anchor_poses, num_anchors)

    # 4. Initial Guess
    x0 = np.array([p for p in snapshot['nodes']]).flatten()

    # 5. Pre-Compute Sparse Structure Matrix Blueprint
    jac_rows = []
    jac_cols = []

    # Structural index map for Edge constraints
    for idx in range(num_edges):
        r_start = 3 * idx
        f_col = 3 * from_ids[idx]
        t_col = 3 * to_ids[idx]
        # Ji indices (row-major alignment)
        for r in range(3):
            for c in range(3):
                jac_rows.append(r_start + r)
                jac_cols.append(f_col + c)
        # Jj indices (row-major alignment)
        for r in range(3):
            for c in range(3):
                jac_rows.append(r_start + r)
                jac_cols.append(t_col + c)
                
    # Structural index map for Anchor constraints
    for idx in range(num_anchors):
        r_start = 3 * num_edges + 3 * idx
        a_col = 3 * anchored_ids[idx]
        for r in range(3):
            jac_rows.append(r_start + r)
            jac_cols.append(a_col + r)
            
    jac_rows = np.array(jac_rows, dtype=int)
    jac_cols = np.array(jac_cols, dtype=int)

    # 5. Define Analytical Jacobian Callback
    def jac_func(x):
        return jacobian_matrix(x, n, from_ids, to_ids, sqrt_omegas, jac_rows, jac_cols, num_edges, num_anchors)

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
        snapshot['nodes'][i][0] = optimized_nodes[i, 0]
        snapshot['nodes'][i][1] = optimized_nodes[i, 1]
        snapshot['nodes'][i][2] = norm_angle_vec(optimized_nodes[i, 2])


def optimize_cpp(snapshot : dict[str, Any], num_iterations : int = 10, window_size : int = 0):
    """
    Zero-copy fast path passing data directly to C++ Eigen engine.

    Args:
        snapshot (dict[str, Any]): Snapshot dictionary containing 'nodes' and 'edges'.
        num_iterations (int, optional): Optimization iterations. Defaults to 10.
        window_size (int, optional): Size of the active sliding window. Defaults to 0.
    """
    n : int         = len(snapshot['nodes'])
    num_edges : int = len(snapshot['edges'])
    huber_k : float = 2.0

    # 1. Get Flattened Nodes: (N, 3) Float64 Array
    nodes : np.ndarray = snapshot['nodes']

    # 2. Flatten Edges -> (E, 14) Float64 Array
    edges_flat : np.ndarray = np.zeros((num_edges, 14), dtype=np.float64)
    for i, (f_id, t_id, m, omega) in enumerate(snapshot['edges']):
        edges_flat[i, 0] = f_id
        edges_flat[i, 1] = t_id
        edges_flat[i, 2:5] = m
        edges_flat[i, 5:14] = omega.flatten()

    # 3. Anchors
    anchors : list = []
    if window_size > 0 and n > window_size:
        anchors = list(range(n - window_size))
    else:
        anchors = [0]

    # 4. Execute C++ Engine (Releases the GIL, executes at bare-metal speed)
    optimized_nodes = succulence_cpp_optimizer.optimize_graph_cpp(
        nodes, edges_flat, anchors, num_iterations, huber_k, 1e6
    )

    # 5. Write back to Graph
    for i in range(n):
        snapshot['nodes'][i][0] = optimized_nodes[i, 0]
        snapshot['nodes'][i][1] = optimized_nodes[i, 1]
        snapshot['nodes'][i][2] = optimized_nodes[i, 2]


def optimize(snapshot : dict[str, Any], num_iterations : int = 10, window_size : int = 0, c_engine : str = ''):
    """
    Master Router Function directing graph data to the selected optimization backend.
    
    Args:
        snapshot (dict[str, Any]): Snapshot dictionary containing 'nodes' and 'edges'.
        num_iterations (int, optional): Optimization iterations. Defaults to 10.
        window_size (int, optional): Size of the active sliding window. Defaults to 0.
        c_engine (str, optional): Target backend engine ('cpp', 'full', or fallback). Defaults to ''.
    """
    if len(snapshot['nodes']) < 2 or len(snapshot['edges']) == 0:
        return
    
    c_flag : str = str(c_engine).lower()

    if c_flag == 'cpp' and HAS_CPP_ENGINE:
        optimize_cpp(snapshot, num_iterations, window_size)
    elif c_flag in ['cpp', 'full']:
        optimize_c(snapshot, num_iterations, window_size)
    else:
        optimize_lm(snapshot, num_iterations, window_size)