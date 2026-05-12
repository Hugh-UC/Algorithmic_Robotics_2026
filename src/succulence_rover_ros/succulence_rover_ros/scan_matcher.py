"""
Correlation-Based Scan Matching

The scan matcher is used in the SLAM node to provide relative pose 
estimates between consecutive laser scans (scan-to-scan) and the current
laser scan against the map (scan-to-map).

This module aligns consecutive laser scans by searching over candidate
relative poses and scoring each one using grid correlation. It returns
the best-scoring pose along with a covariance estimated from the score
surface's Hessian.

The approach is from Olson (2009) "Real-Time Correlative Scan Matching",
simplified for teaching. It's brute-force (O(n^3) over the search space)
— not elegant, but simple, reliable, and guaranteed to find the global
optimum within the search window.

Extension: A course-to-fine search strategy is implemented to speed up
        the search while maintaining good accuracy.

Usage: The ScanMatcher class is instantiated in the SLAM node and called
        with new laser scans to get relative pose estimates for the SLAM
        factor graph.
"""

import numpy as np
from typing import Tuple, Callable
from scipy.ndimage import maximum_filter


class ScanMatcher:
    """
    Correlation-based scan matcher.

    Aligns two laser scans by searching over candidate relative poses
    and scoring each one using grid correlation.
    """
    def __init__(self,
                 search_x: float,
                 search_y: float,
                 search_theta: float,
                 resolution_x: float,
                 resolution_y: float,
                 resolution_theta: float,
                 local_grid_size: int,
                 local_grid_resolution: float,
                 min_score: float):
        self.search_x = search_x
        self.search_y = search_y
        self.search_theta = search_theta
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_theta = resolution_theta
        self.local_grid_size = local_grid_size
        self.local_grid_resolution = local_grid_resolution
        self.min_score = min_score


    # ========================================================================
    # Build Local Occupancy Grid
    # ========================================================================
    def _build_local_grid(self, scan_points: np.ndarray) -> np.ndarray:
        """
        Rasterise scan points into a local occupancy grid for fast correlation.

        The grid is centred at the origin. Each scan point is placed into its
        corresponding cell. After rasterisation, dilate the grid by 1 cell
        (3x3 maximum filter) to tolerate small alignment errors.

        Args:
            scan_points: Nx2 array of (x, y) points in local frame

        Returns:
            2D float32 grid (1.0 where scan points land, 0.0 elsewhere)

        Algorithm:
            1. grid = zeros(local_grid_size x local_grid_size)
            2. offset = local_grid_size // 2  (centre of grid)
            3. For each point (x, y):
               col = int(x / local_grid_resolution) + offset
               row = int(y / local_grid_resolution) + offset
               if in bounds: grid[row, col] = 1.0
            4. grid = maximum_filter(grid, size=3)  (dilate for tolerance)
            5. return grid

        Hints:
            - You can vectorise steps 3-4 with NumPy for speed, or use a loop
            - from scipy.ndimage import maximum_filter is already imported
            - ~10-15 lines of code
        """
        grid = np.zeros((self.local_grid_size, self.local_grid_size), dtype=np.float32)
        offset = self.local_grid_size // 2

        # Vetorised version:
        if len(scan_points) > 0:
            # Vectorised conversion to grid coordinates
            cols = (scan_points[:, 0] / self.local_grid_resolution).astype(int) + offset
            rows = (scan_points[:, 1] / self.local_grid_resolution).astype(int) + offset

            # Create a mask for points that fall within the grid bounds
            valid_mask = (rows >= 0) & (rows < self.local_grid_size) & \
                         (cols >= 0) & (cols < self.local_grid_size)
            
            # Assign 1.0 only to valid coordinates
            grid[rows[valid_mask], cols[valid_mask]] = 1.0

        # Non-vectorised version:
        '''
        for x, y in scan_points:
            col = int(x / self.local_grid_resolution) + offset
            row = int(y / self.local_grid_resolution) + offset

            if 0 <= row < self.local_grid_size and 0 <= col < self.local_grid_size:
                grid[row, col] = 1.0
        '''

        # Dilate for tolerance (1 cell radius → 3x3 maximum filter)
        grid = maximum_filter(grid, size=3)

        return grid


    # ========================================================================
    # Score Alignment
    # ========================================================================
    def _score_alignment(self, grid: np.ndarray, scan_points: np.ndarray, pose: np.ndarray) -> float:
        """
        Score how well scan_points align with the reference grid
        when transformed by the candidate pose.

        Args:
            grid:        Reference scan's local occupancy grid
            scan_points: Nx2 array of new scan points (local frame)
            pose:        Candidate relative pose [dx, dy, dtheta]

        Returns:
            Correlation score (count of overlapping points)

        Algorithm:
            1. dx, dy, dtheta = pose
            2. c, s = cos(dtheta), sin(dtheta)
            3. For each point (px, py):
               px' = c*px - s*py + dx     (rotate then translate)
               py' = s*px + c*py + dy
               col = int(px' / resolution) + offset
               row = int(py' / resolution) + offset
               if in bounds and grid[row, col] > 0: score += 1
            4. return score

        Hints:
            - Precompute cos/sin once, not per point
            - You can vectorise with NumPy or use a loop
            - ~10-15 lines of code
        """
        # Extract pose components
        dx, dy, dtheta = pose
        
        # Compute cos/sin once
        c, s = np.cos(dtheta), np.sin(dtheta)

        # Vectorise the transformation of scan points
        px = scan_points[:, 0]
        py = scan_points[:, 1]
        
        px_prime = c * px - s * py + dx
        py_prime = s * px + c * py + dy
        
        # Convert to grid coordinates
        offset = self.local_grid_size // 2
        cols = (px_prime / self.local_grid_resolution).astype(int) + offset
        rows = (py_prime / self.local_grid_resolution).astype(int) + offset
        
        # Find points within the grid bounds
        valid_mask = (rows >= 0) & (rows < self.local_grid_size) & \
                     (cols >= 0) & (cols < self.local_grid_size)
        
        # Count valid points in occupied cell (> 0)
        hits = grid[rows[valid_mask], cols[valid_mask]] > 0
        
        # Return total score
        return float(np.sum(hits))


    # ========================================================================
    # Universal Searcher (scan-to-scan and scan-to-map)
    # ========================================================================
    def _coarse_to_fine_search(self, score_fn: Callable[[np.ndarray], float], initial_guess: np.ndarray) -> Tuple[np.ndarray, dict, Tuple[int, int, int], float]:
        """
        Universal Coarse-to-Fine grid search.

        Args:
            score_fn (Callable[[np.ndarray], float]): _description_
            initial_guess (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, dict, Tuple[int, int, int], float]: _description_
        """
        # --- phase 1 | coarse ---
        cx_step, cy_step, ct_step = self.resolution_x * 4.0, self.resolution_y * 4.0, self.resolution_theta * 4.0
        
        cx_vals = np.arange(-self.search_x, self.search_x + 1e-6, cx_step)
        cy_vals = np.arange(-self.search_y, self.search_y + 1e-6, cy_step)
        ct_vals = np.arange(-self.search_theta, self.search_theta + 1e-6, ct_step)

        best_coarse_score = -1.0
        best_coarse_offset = np.zeros(3)

        for dx in cx_vals:
            for dy in cy_vals:
                for dt in ct_vals:
                    candidate = initial_guess + np.array([dx, dy, dt])
                    score = score_fn(candidate)
                    if score > best_coarse_score:
                        best_coarse_score = score
                        best_coarse_offset = np.array([dx, dy, dt])

        # --- phase 2 | fine ---
        fine_center = initial_guess + best_coarse_offset
        fx_vals = np.arange(-cx_step, cx_step + 1e-6, self.resolution_x)
        fy_vals = np.arange(-cy_step, cy_step + 1e-6, self.resolution_y)
        ft_vals = np.arange(-ct_step, ct_step + 1e-6, self.resolution_theta)

        scores = {}
        best_fine_score = -1.0
        best_pose = fine_center.copy()
        best_idx = (0, 0, 0)

        for ix, dx in enumerate(fx_vals):
            for iy, dy in enumerate(fy_vals):
                for it, dt in enumerate(ft_vals):
                    candidate = fine_center + np.array([dx, dy, dt])
                    score = score_fn(candidate)
                    scores[(ix, iy, it)] = score
                    if score > best_fine_score:
                        best_fine_score, best_pose, best_idx = score, candidate, (ix, iy, it)

        return best_pose, scores, best_idx, best_fine_score


    # ========================================================================
    # Scan-to-Scan Matching
    # ========================================================================
    def match(self, scan_ref: np.ndarray, scan_new: np.ndarray, initial_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Match two scans using correlation-based grid search.

        Finds the relative pose that best aligns scan_new to scan_ref by
        exhaustively searching candidate poses around initial_guess.

        Args:
            scan_ref:      Nx2 reference scan points in local frame
            scan_new:      Mx2 new scan points in local frame
            initial_guess: Initial relative pose estimate [dx, dy, dtheta]

        Returns:
            best_pose:  Best relative pose [dx, dy, dtheta]
            covariance: 3x3 covariance matrix of the match
            score:      Normalised score in [0, 1] (0 if match rejected)

        Algorithm:
            1. ref_grid = _build_local_grid(scan_ref)
            2. Generate search grid with np.arange():
               x_values = [guess_x - search_x ... guess_x + search_x]
               y_values = [guess_y - search_y ... guess_y + search_y]
               theta_values = [guess_t - search_t ... guess_t + search_t]
            3. For each (ix, x), (iy, y), (it, theta):
               candidate = [x, y, theta]
               score = _score_alignment(ref_grid, scan_new, candidate)
               scores[(ix, iy, it)] = score
               Track best score + pose + indices
            4. normalised_score = best_score / len(scan_new)
            5. If normalised_score < min_score: return (initial_guess, default_cov, 0.0)
            6. covariance = _estimate_covariance_from_hessian(scores, best_idx, ...)
            7. Return (best_pose, covariance, normalised_score)
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        if len(scan_ref) == 0 or len(scan_new) == 0:
            return initial_guess.copy(), default_cov, 0.0

        # Step 1: Build local occupancy grid from the reference scan
        ref_grid = self._build_local_grid(scan_ref)

        # Step 2: Create lambda wrapper to pass scoring method
        score_fn = lambda pose: self._score_alignment(ref_grid, scan_new, pose)

        # Step 3: Coarse-to-fine search over candidate poses
        best_pose, scores, best_idx, best_score = self._coarse_to_fine_search(score_fn, initial_guess)
        
        # Step 4: Normalise score and check against threshold
        normalised_score = best_score / len(scan_new)

        if normalised_score < self.min_score:
            return initial_guess.copy(), default_cov, 0.0 

        covariance = self._estimate_covariance_from_hessian(
            scores, best_idx,
            self.resolution_x, self.resolution_y, self.resolution_theta
        )

        return best_pose, covariance, normalised_score

    # ========================================================================
    # Scan-to-Map Matching (Extension)
    # ========================================================================
    def match_to_map(self, global_grid: np.ndarray, map_origin_x: float, map_origin_y: float, map_resolution: float,
                     scan_new: np.ndarray, initial_guess_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Aligns a live scan against the global SLAM occupancy grid.

        Args:
            global_grid (np.ndarray): _description_
            map_origin_x (float): _description_
            map_origin_y (float): _description_
            map_resolution (float): _description_
            scan_new (np.ndarray): _description_
            initial_guess_global (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: _description_
        """
        default_cov = np.diag([0.1, 0.1, 0.05])
        if len(scan_new) == 0:
            return initial_guess_global.copy(), default_cov, 0.0

        # Step 1: Define specific scoring rules for global scan-to-map using real-world coordinates
        def score_global(pose: np.ndarray) -> float:
            c, s = np.cos(pose[2]), np.sin(pose[2])

            # rotate and translate points into global map frame
            px_prime = c * scan_new[:, 0] - s * scan_new[:, 1] + pose[0]
            py_prime = s * scan_new[:, 0] + c * scan_new[:, 1] + pose[1]
            
            cols = ((px_prime - map_origin_x) / map_resolution).astype(int)
            rows = ((py_prime - map_origin_y) / map_resolution).astype(int)
            
            h, w = global_grid.shape
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            hits = global_grid[rows[valid], cols[valid]] > 0
            return float(np.sum(hits))

        # Step 2: Perform coarse-to-fine search
        best_pose, scores, best_idx, best_score = self._coarse_to_fine_search(score_global, initial_guess_global)

        # Step 3: Normalise score and check against threshold
        normalised_score = best_score / len(scan_new)
        if normalised_score < self.min_score:
            return initial_guess_global.copy(), default_cov, 0.0

        cov = self._estimate_covariance_from_hessian(
            scores, best_idx,
            self.resolution_x, self.resolution_y, self.resolution_theta
        )

        return best_pose, cov, normalised_score
    

    # ========================================================================
    # Estimate Covariance from Hessian
    # ========================================================================
    def _estimate_covariance_from_hessian(self,
                                          scores: dict,
                                          best_idx: Tuple[int, int, int],
                                          step_x: float,
                                          step_y: float,
                                          step_theta: float) -> np.ndarray:
        """
        Estimate match covariance from the Hessian of the score surface.

        Args:
            scores:     Dict mapping (ix, iy, it) tuples to score values
            best_idx:   (ix, iy, it) of the peak score
            step_x:     Search step in x (metres)
            step_y:     Search step in y (metres)
            step_theta: Search step in theta (radians)

        Returns:
            3x3 covariance matrix
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        f0 = scores.get(best_idx, 0.0)
        if f0 == 0.0:
            return default_cov

        steps = [step_x, step_y, step_theta]
        H = np.zeros((3, 3))

        # Diagonal elements: H[i,i] = (f(+step) - 2*f0 + f(-step)) / step_i^2
        for i in range(3):
            idx_plus = list(best_idx)
            idx_minus = list(best_idx)
            idx_plus[i] += 1
            idx_minus[i] -= 1

            f_plus = scores.get(tuple(idx_plus), 0.0)
            f_minus = scores.get(tuple(idx_minus), 0.0)

            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        # Off-diagonal elements: H[i,j] = (f++ - f+- - f-+ + f--) / (4*si*sj)
        for i in range(3):
            for j in range(i + 1, 3):
                idx_pp = list(best_idx)
                idx_pm = list(best_idx)
                idx_mp = list(best_idx)
                idx_mm = list(best_idx)

                idx_pp[i] += 1; idx_pp[j] += 1
                idx_pm[i] += 1; idx_pm[j] -= 1
                idx_mp[i] -= 1; idx_mp[j] += 1
                idx_mm[i] -= 1; idx_mm[j] -= 1

                f_pp = scores.get(tuple(idx_pp), 0.0)
                f_pm = scores.get(tuple(idx_pm), 0.0)
                f_mp = scores.get(tuple(idx_mp), 0.0)
                f_mm = scores.get(tuple(idx_mm), 0.0)

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                H[j, i] = H[i, j]

        # Covariance = (-H)^{-1}  (negate because H is concave at a maximum)
        neg_H = -H

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(neg_H)
        if np.any(eigenvalues <= 1e-6):
            return default_cov

        try:
            covariance = np.linalg.inv(neg_H)
        except np.linalg.LinAlgError:
            return default_cov

        # Sanity check
        cov_eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(cov_eigenvalues <= 0):
            return default_cov

        return covariance


# ============================================================================
# Helper function
# ============================================================================
def scans_from_ranges(ranges: np.ndarray, angle_min: float,
                      angle_increment: float, min_range: float = 0.1,
                      max_range: float = 12.0,
                      lidar_yaw_offset: float = 0.0) -> np.ndarray:
    """
    Convert laser scan ranges to (x, y) points in the robot's local frame.

    Converts raw LaserScan ranges into the Nx2 point arrays that the ScanMatcher expects.

    Args:
        ranges:           Array of laser range measurements (metres)
        angle_min:        Start angle of scan (radians)
        angle_increment:  Angular step between beams (radians)
        min_range:        Minimum valid range (metres)
        max_range:        Maximum valid range (metres)
        lidar_yaw_offset: Lidar yaw rotation relative to base_link (radians)

    Returns:
        Nx2 array of (x, y) points in robot's local frame
    """
    points = []
    for i, r in enumerate(ranges):
        if np.isnan(r) or r < min_range or r > max_range:
            continue
        beam_angle = lidar_yaw_offset + angle_min + i * angle_increment
        x = r * np.cos(beam_angle)
        y = r * np.sin(beam_angle)
        points.append([x, y])

    if len(points) == 0:
        return np.empty((0, 2))
    return np.array(points)
