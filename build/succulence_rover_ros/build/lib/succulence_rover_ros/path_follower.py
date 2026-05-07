"""
Pure Pursuit Path Follower (no ROS dependencies)

Classic pure pursuit: pick the farthest point on the path within `lookahead`
metres of the robot, steer toward it. Linear speed ramps down as the robot
approaches the final waypoint.
"""

import numpy as np
from typing import List, Tuple, Optional


def _normalize_angle(theta: float) -> float:
    while theta > np.pi:
        theta -= 2.0 * np.pi
    while theta < -np.pi:
        theta += 2.0 * np.pi
    return theta


class PurePursuit:
    """Pure-pursuit controller that outputs (v, w) given pose and an (x, y) path."""

    def __init__(self,
                 lookahead: float,
                 max_linear_v: float,
                 max_angular_v: float,
                 goal_tolerance: float):
        self.lookahead = lookahead
        self.max_linear_v = max_linear_v
        self.max_angular_v = max_angular_v
        self.goal_tolerance = goal_tolerance

    def compute_cmd(self,
                    pose: np.ndarray,
                    path: List[Tuple[float, float]]) -> Tuple[float, float, bool]:
        """
        Returns (linear_v, angular_w, arrived).
        `arrived` is True once the robot is within goal_tolerance of the last waypoint.
        """
        if not path:
            return 0.0, 0.0, False

        x, y, theta = pose
        goal = path[-1]
        dgoal = np.hypot(goal[0] - x, goal[1] - y)

        if dgoal < self.goal_tolerance:
            return 0.0, 0.0, True

        target = self._select_lookahead(pose, path)

        dx = target[0] - x
        dy = target[1] - y

        heading = np.arctan2(dy, dx)
        heading_error = _normalize_angle(heading - theta)

        # Cap forward speed as we approach the goal, and slow down during
        # sharp turns so the controller doesn't swing wide.
        v = min(self.max_linear_v, dgoal)
        v *= max(0.0, np.cos(heading_error))
        v = max(0.0, v)

        # Curvature-limited speed cap. Pure pursuit demands
        #   w = 2 * v * sin(err) / lookahead
        # If that exceeds max_angular_v, the controller saturates and the
        # rover overshoots (drunk wobble). Instead, scale v down so the
        # demanded w fits inside the angular cap. Net effect: rover stays
        # at full speed on straight stretches and automatically slows
        # through corners.
        w_demanded = 2.0 * v * np.sin(heading_error) / max(self.lookahead, 1e-3)
        if abs(w_demanded) > self.max_angular_v and abs(w_demanded) > 1e-6:
            scale = self.max_angular_v / abs(w_demanded)
            v *= scale
            w_demanded *= scale
        w = float(np.clip(w_demanded, -self.max_angular_v, self.max_angular_v))

        # When heading error is large, spin in place rather than moving forward.
        if abs(heading_error) > np.pi / 3.0:
            v = 0.0
            w = float(np.clip(
                2.0 * heading_error, -self.max_angular_v, self.max_angular_v))

        return v, w, False

    def _select_lookahead(self,
                          pose: np.ndarray,
                          path: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Pick the farthest point along the path still within the lookahead radius."""
        x, y = pose[0], pose[1]

        # Find the nearest point on the path as a starting index
        best_idx = 0
        best_d = float('inf')
        for i, (px, py) in enumerate(path):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_d = d
                best_idx = i

        target = path[-1]
        for i in range(best_idx, len(path)):
            px, py = path[i]
            if np.hypot(px - x, py - y) >= self.lookahead:
                target = (px, py)
                break

        return target
