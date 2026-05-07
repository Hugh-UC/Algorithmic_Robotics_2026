"""
A* Path Planner (no ROS dependencies)

8-connected grid search with octile heuristic. Operates on a 2D bool array
where True = blocked (obstacle or inflated obstacle) and False = free.

Student task (1 TODO):
  - TODO #1: octile heuristic

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


def inflate_obstacles(grid: np.ndarray, radius_cells: int,
                      occupancy_threshold: int,
                      treat_unknown_as_obstacle: bool) -> np.ndarray:
    """
    Return a bool grid of 'blocked' cells after inflating obstacles by
    radius_cells. Provided — you do not need to modify this.

    Unknown cells (-1) are blocked or free depending on the flag.
    Inflation uses square dilation for simplicity.
    """
    blocked = grid >= occupancy_threshold
    if treat_unknown_as_obstacle:
        blocked |= grid < 0

    if radius_cells <= 0:
        return blocked

    h, w = blocked.shape
    out = np.zeros_like(blocked)
    rows, cols = np.where(blocked)
    for r, c in zip(rows, cols):
        r0 = max(0, r - radius_cells)
        r1 = min(h, r + radius_cells + 1)
        c0 = max(0, c - radius_cells)
        c1 = min(w, c + radius_cells + 1)
        out[r0:r1, c0:c1] = True
    return out


def astar_search(blocked: np.ndarray,
                 start: Cell,
                 goal: Cell) -> Optional[List[Cell]]:
    """
    Run A* over an 8-connected grid.

    Args:
        blocked: 2D bool array — True where the robot cannot pass.
        start:   (row, col) start cell (must be unblocked).
        goal:    (row, col) goal cell (must be unblocked).

    Returns:
        List of (row, col) cells from start to goal inclusive, or None
        if unreachable.
    """
    h, w = blocked.shape

    # --- PROVIDED: Boundary and trivial-case checks ---
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if blocked[start] or blocked[goal]:
        return None
    if start == goal:
        return [start]

    # --- PROVIDED: Initialise A*'s data structures ---
    #   open_heap: priority queue of (f, h, counter, cell). The counter
    #              breaks ties so equal-f cells pop in insertion order.
    #   g_score:   best-known cost from start to each cell so far.
    #   came_from: predecessor along the best-known path (for path
    #              reconstruction once the goal is popped).
    #   closed:    cells that have already been expanded.
    open_heap: List[Tuple[float, float, int, Cell]] = []
    counter = 0
    g_score = {start: 0.0}
    came_from: dict = {}
    closed: set = set()

    h0 = _octile(start, goal)
    heapq.heappush(open_heap, (h0, h0, counter, start))

    while open_heap:
        # --- PROVIDED: Pop the lowest-f cell, skip stale heap entries ---
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        # --- PROVIDED: Goal check + path reconstruction via came_from ---
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

        # --- PROVIDED: Iterate over 8 neighbours with the defensive checks
        #     (bounds, blocked, closed, corner-cutting). These aren't
        #     interesting algorithmically — they just keep the search safe.
        for dr, dc, step in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc]:
                continue
            # Stop the rover squeezing diagonally through a 1-cell gap.
            if dr != 0 and dc != 0:
                if blocked[cr + dr, cc] and blocked[cr, cc + dc]:
                    continue
            neighbor = (nr, nc)
            if neighbor in closed:
                continue

            # --- PROVIDED: Edge relaxation (the heart of A*) ---
            # If reaching `neighbor` via `current` is cheaper than the
            # best route we've seen so far, record the new route and
            # re-queue with f = g + h. That `f = g + h` line is what
            # makes this A* and not Dijkstra: the heuristic biases the
            # heap toward the goal so we expand far fewer cells.
            tentative_g = cg + step
            if tentative_g < g_score.get(neighbor, np.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                hscore = _octile(neighbor, goal)
                f = tentative_g + hscore
                counter += 1
                heapq.heappush(open_heap, (f, hscore, counter, neighbor))

    return None
