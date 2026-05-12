"""
A* Path Planner (no ROS dependencies)

8-connected grid search with octile heuristic. Operates on a 2D bool array
where True = blocked (obstacle or inflated obstacle) and False = free.

References:
    - Hart, Nilsson, Raphael (1968) — original A* paper
    - Lecture 12: Path Planning
"""

import heapq
import numpy as np
from typing import List, Optional, Tuple

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
def _octile(a: Cell, b: Cell) -> float:
    """
    Octile distance between two grid cells — admissible heuristic for an
    8-connected grid where cardinal moves cost 1 and diagonal moves cost sqrt(2).

    For two cells with row/col deltas dr and dc:
        h = (dr + dc) + (sqrt(2) - 2) * min(dr, dc)

    Intuition: take min(dr, dc) diagonal steps, then |dr - dc| straight steps.

    Args:
        a, b: (row, col) cells.

    Returns:
        Estimated cost (lower bound on true path cost) from a to b.
    """
    dr : int = abs(a[0] - b[0])
    dc : int = abs(a[1] - b[1])

    h : float = float((dr + dc) + (_SQRT2 - 2) * min(dr, dc))
    return h


def inflate_obstacles(grid: np.ndarray, radius_cells: float,
                      occupancy_threshold: int,
                      treat_unknown_as_obstacle: bool, inflation_weight: float = 5.0) -> np.ndarray:
    """
    Returns a float grid where np.inf is a solid wall, 0.0 is open space, 
    and values in between act as a penalty gradient pushing the robot away from walls.

    OLD: "Return a bool grid of 'blocked' cells after inflating obstacles by
    radius_cells. Provided — you do not need to modify this.

    Unknown cells (-1) are blocked or free depending on the flag.
    Inflation uses square dilation for simplicity.
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
        return cost_map
    
    # calculate integer boundary for array slicing
    bound_r = int(np.ceil(radius_cells))

    rows, cols = np.where(blocked)
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
                    
    return cost_map


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

    h0 = _octile(start, goal)
    heapq.heappush(open_heap, (h0, h0, counter, start))

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
                hscore = _octile(neighbor, goal)

                # multiply heuristic by epsilon for weighted A*
                f = tentative_g + (hscore * epsilon)

                counter += 1
                heapq.heappush(open_heap, (f, hscore, counter, neighbor))

    return None
