"""
A* Path Planner (no ROS dependencies)

8-connected grid search with octile heuristic. Operates on a 2D bool array
where True = blocked (obstacle or inflated obstacle) and False = free.

References:
    - Hart, Nilsson, Raphael (1968) — original A* paper
    - Lecture 12: Path Planning
"""

import time
import heapq
import numpy as np
from typing import List, Optional, Tuple
from scipy.ndimage import distance_transform_edt

Cell = Tuple[int, int]

# 8-connected step costs: 1 for cardinal, sqrt(2) for diagonal.
_SQRT2 = np.sqrt(2.0)
_NEIGHBORS = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, _SQRT2),
    (-1,  1, _SQRT2),
    ( 1, -1, _SQRT2),
    ( 1,  1, _SQRT2),
]


# ============================================================================
# 1: Octile heuristic
# ============================================================================
def _octile(a: Cell, b: Cell, start: Optional[Cell] = None) -> float:
    """
    Octile distance between two grid cells — admissible heuristic for an
    8-connected grid where cardinal moves cost 1 and diagonal moves cost sqrt(2).
    With a cross-product tie-breaker to force straight lines.

    For two cells with row/col deltas dr and dc:
        h = (dr + dc) + (sqrt(2) - 2) * min(dr, dc)

    Intuition: take min(dr, dc) diagonal steps, then |dr - dc| straight steps.

    Args:
        a, b: (row, col) cells.
        start: Optional (row, col) start cell for tie-breaking. If provided,
               the heuristic will prefer paths that are more collinear with
               the start-goal line, which can help reduce A*'s tendency to
               explore large open areas in a circular pattern.

    Returns:
        Estimated cost (lower bound on true path cost) from a to b.
    """
    dr : int = abs(a[0] - b[0])
    dc : int = abs(a[1] - b[1])

    # standard octile heuristic
    h : float = float((dr + dc) + (_SQRT2 - 2) * min(dr, dc))

    # cross-product tie-breaker
    if start is not None:
        dx1 = a[0] - b[0]
        dy1 = a[1] - b[1]
        dx2 = start[0] - b[0]
        dy2 = start[1] - b[1]
        cross = abs(dx1 * dy2 - dx2 * dy1)
        h += cross * 0.001      # add a tiny penalty

    return h


def inflate_obstacles_loops(grid: np.ndarray,
                            radius_cells: float,
                            occupancy_threshold: int,
                            treat_unknown_as_obstacle: bool, 
                            inflation_weight: float = 5.0) -> tuple[np.ndarray, int]:
    """
    Standard inflation using triple-nested Python loops. 
    Best for small ROI windows where readability and standard math are preferred.

    Args:
        grid (np.ndarray): The raw 2D occupancy grid from SLAM (int8).
        radius_cells (float): Radius to inflate obstacles by, in grid cells.
        occupancy_threshold (int): Value (0-100) above which a cell is blocked.
        treat_unknown_as_obstacle (bool): If True, treats -1 (unmapped) as a wall.
        inflation_weight (float, optional): Maximum cost penalty at the wall. Defaults to 5.0.

    Returns:
        np.ndarray: A 32-bit float costmap where np.inf is a wall and 0.0 is free.
    """
    blocked = grid >= occupancy_threshold
    if treat_unknown_as_obstacle:
        blocked |= grid < 0

    h, w = grid.shape
    # initialize everything to 0.0 (free space)
    cost_map = np.zeros((h, w), dtype=np.float32)
    
    # mark solid obstacles as mathematically impassable
    cost_map[blocked] = np.inf

    if radius_cells <= 0:
        return cost_map, 0
    
    # calculate integer boundary for array slicing
    bound_r = int(np.ceil(radius_cells))
    rows, cols = np.where(blocked)
    # --- LOGGING: Track the scale of work ---
    num_blocked_cells : int = int(len(rows))

    for r, c in zip(rows, cols):
        r0 = max(0, r - bound_r)
        r1 = min(h, r + bound_r + 1)
        c0 = max(0, c - bound_r)
        c1 = min(w, c + bound_r + 1)

        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if cost_map[rr, cc] == np.inf:
                    continue
                
                dist = np.hypot(rr - r, cc - c)
                if dist <= radius_cells:
                    penalty = inflation_weight * (1.0 - (dist / radius_cells))
                    cost_map[rr, cc] = max(cost_map[rr, cc], penalty)

    return cost_map, num_blocked_cells


def inflate_obstacles_vectors(grid: np.ndarray,
                                radius_cells: float,
                                occupancy_threshold: int,
                                treat_unknown_as_obstacle: bool, 
                                inflation_weight: float = 5.0) -> np.ndarray:
    """
    Truly Global Inflation using C++ optimized Euclidean Distance Transform (EDT).
    Handles millions of cells in milliseconds. Recommended for full-map updates.

    Args:
        grid (np.ndarray): The raw 2D occupancy grid from SLAM (int8).
        radius_cells (float): Radius to inflate obstacles by, in grid cells.
        occupancy_threshold (int): Value (0-100) above which a cell is blocked.
        treat_unknown_as_obstacle (bool): If True, treats -1 (unmapped) as a wall.
        inflation_weight (float, optional): Maximum cost penalty at the wall. Defaults to 5.0.

    Returns:
        np.ndarray: A 32-bit float costmap with safety gradients.
    """
    # 1. Identify obstacles
    blocked = grid >= occupancy_threshold
    if treat_unknown_as_obstacle:
        blocked |= (grid < 0)
    
    cost_map = np.zeros(grid.shape, dtype=np.float32)
    cost_map[blocked] = np.inf

    if radius_cells <= 0:
        return cost_map

    # 2. Distance Transform, calculates distance from every cell to nearest blocked cell
    dist_to_obstacle = distance_transform_edt(~blocked)

    # 3. Apply Gradient to all cells at once
    mask = (dist_to_obstacle > 0) & (dist_to_obstacle <= radius_cells)
    cost_map[mask] = inflation_weight * (1.0 - (dist_to_obstacle[mask] / radius_cells))

    return cost_map


def inflate_obstacles(name: str, method : str,
                      grid: np.ndarray,
                      radius_cells: float,
                      occupancy_threshold: int,
                      treat_unknown_as_obstacle: bool,
                      inflation_weight: float = 5.0) -> np.ndarray:
    """
    Wrapper function to generate a costmap using either local (loop) or global (EDT) methods.

    Args:
        method (str): Choose 'local' for standard loops or 'global' for optimized EDT.
        grid (np.ndarray): The 2D input grid.
        radius_cells (float): Inflation radius in cells.
        occupancy_threshold (int): Threshold for identifying obstacles.
        treat_unknown_as_obstacle (bool): Flag for unknown space handling.
        inflation_weight (float, optional): Penalty strength. Defaults to 5.0.

    Returns:
        np.ndarray: The resulting float32 costmap.

    Raises:
        ValueError: If an unsupported method string is provided.
    """
    # Start clock for inflation computation timing
    t_start = time.time()

    if method == 'python':
        cost_map, blocked = inflate_obstacles_loops(grid, radius_cells, occupancy_threshold, treat_unknown_as_obstacle, inflation_weight)

        # Log inflation computation time
        t_end = time.time()
        print(f"[DEBUG PLANNER] {name.title()} Costmap Inflated {blocked} obstacles (Python Loops). Time: {t_end - t_start:.4f}s")

        return cost_map
        

    if method == 'c++':
        cost_map = inflate_obstacles_vectors(grid, radius_cells, occupancy_threshold, treat_unknown_as_obstacle, inflation_weight)
        
        # Log inflation computation time
        t_end = time.time()
        print(f"[DEBUG PLANNER] {name.title()} Costmap Inflation (Vectorized): {t_end - t_start:.4f}s")

        return cost_map

    raise ValueError(f"Invalid inflation method: {method}. Use 'python' or 'c++'.")


def astar_search(cost_map: np.ndarray,
                 start: Cell,
                 goal: Cell,
                 epsilon: float = 1.0) -> Optional[List[Cell]]:
    """
    Run A* over an 8-connected grid.

    Args:
        cost_map: 2D float array — np.inf where the robot cannot pass, 0.0 where it can.
        start:   (row, col) start cell (must be unblocked).
        goal:    (row, col) goal cell (must be unblocked). 
        epsilon: Heuristic weight — higher values make A* more greedy and
                 faster to run but less likely to find a solution.
    Returns:
        List of (row, col) cells from start to goal inclusive, or None
        if unreachable.
    """
    h, w = cost_map.shape

    # --- boundary and trivial-case checks ---
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if cost_map[start] == np.inf or cost_map[goal] == np.inf:
        return None
    if start == goal:
        return [start]


    open_heap: List[Tuple[float, float, int, Cell]] = []
    counter = 0
    g_score = {start: 0.0}
    came_from: dict = {}
    closed: set = set()

    h0 = _octile(start, goal, start)
    heapq.heappush(open_heap, (h0 * epsilon, h0, counter, start))

    while open_heap:
        # --- pop the lowest-f cell, skip stale heap entries ---
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        # --- goal check + path reconstruction via came_from ---
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)
        cg = g_score[current]
        cr, cc = current

        # --- iterate over 8 neighbours with the defensive checks
        for dr, dc, step in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            
            # read penalty from cost map
            penalty = cost_map[nr, nc]
            if penalty == np.inf:
                continue

            # prevent diagonal squeezing through 1-cell gap
            if dr != 0 and dc != 0:
                if cost_map[cr + dr, cc] == np.inf and cost_map[cr, cc + dc] == np.inf:
                    continue

            neighbor = (nr, nc)
            if neighbor in closed:
                continue

            # --- edge relaxation (the heart of A*) ---
            tentative_g = cg + step + penalty

            if tentative_g < g_score.get(neighbor, np.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                # heuristic estimate from neighbor to goal
                hscore = _octile(neighbor, goal, start)

                # multiply heuristic by epsilon for weighted A*
                f = tentative_g + (hscore * epsilon)

                counter += 1
                heapq.heappush(open_heap, (f, hscore, counter, neighbor))

    return None
