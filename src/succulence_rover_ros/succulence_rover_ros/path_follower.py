"""
Pure Pursuit Path Follower (no ROS dependencies)

Classic pure pursuit: pick the farthest point on the path within `lookahead`
metres of the robot, steer toward it. Linear speed ramps down as the robot
approaches the final waypoint.
"""

import numpy as np
from typing import List, Tuple, Optional


def _normalize_angle(theta : float) -> float:
    """
    _summary_

    Args:
        theta (float): _description_

    Returns:
        float: _description_
    """
    while theta > np.pi:
        theta -= 2.0 * np.pi
    while theta < -np.pi:
        theta += 2.0 * np.pi
    return theta


class PurePursuit:
    """
    Pure-pursuit controller that outputs (v, w) given pose and an (x, y) path.
    """
    def __init__(self,
                 lookahead: float,
                 max_linear_v: float,
                 max_angular_v: float,
                 max_linear_a: float,
                 max_angular_a: float,
                 recovery_spin_v: float,
                 goal_tolerance: float):
        self.lookahead       = lookahead
        self.max_linear_v    = max_linear_v
        self.max_angular_v   = max_angular_v
        self.max_linear_a    = max_linear_a
        self.max_angular_a   = max_angular_a
        self.recovery_spin_v = recovery_spin_v
        self.goal_tolerance  = goal_tolerance

        # Internal kinematic state tracking
        self.current_v: float = 0.0
        self.current_w: float = 0.0


    def compute_cmd(self, pose : np.ndarray, path : List[Tuple[float, float]], dt: float, c_stop: bool = False, e_stop: bool = False) -> Tuple[float, float, bool]:
        """
        Returns (linear_v, angular_w, arrived).
        `arrived` is True once the robot is within goal_tolerance of the last waypoint.
        Applies Kinematic Acceleration limits and Safety Overrides internally.
        """
        # 1. Hard Emergency Brake
        if e_stop:
            self.current_v = 0.0
            self.current_w = 0.0
            return 0.0, 0.0, False

        # 2. No path condition
        if not path:
            self.current_v = 0.0
            self.current_w = 0.0
            return 0.0, 0.0, False

        x, y, theta = pose
        goal = path[-1]
        dgoal = np.hypot(goal[0] - x, goal[1] - y)

        if dgoal < self.goal_tolerance:
            self.current_v = 0.0
            self.current_w = 0.0
            return 0.0, 0.0, True

        target = self._select_lookahead(pose, path)

        dx = target[0] - x
        dy = target[1] - y

        heading = np.arctan2(dy, dx)
        heading_error = _normalize_angle(heading - theta)

        # 3. Cap forward speed as we approach the goal, and slow down during
        # sharp turns so the controller doesn't swing wide.
        v_target = min(self.max_linear_v, dgoal)
        v_target *= max(0.0, np.cos(heading_error))
        v_target = max(0.0, v_target)

        # Curvature-limited speed cap. Pure pursuit demands
        #   w = 2 * v * sin(err) / lookahead
        # If that exceeds max_angular_v, the controller saturates and the
        # rover overshoots (drunk wobble). Instead, scale v down so the
        # demanded w fits inside the angular cap. Net effect: rover stays
        # at full speed on straight stretches and automatically slows
        # through corners.
        w_target = 2.0 * v_target * np.sin(heading_error) / max(self.lookahead, 1e-3)
        if abs(w_target) > self.max_angular_v and abs(w_target) > 1e-6:
            scale = self.max_angular_v / abs(w_target)
            v_target *= scale
            w_target *= scale
            
        w_target = float(np.clip(w_target, -self.max_angular_v, self.max_angular_v))

        # When heading error is large, spin in place rather than moving forward.
        if abs(heading_error) > np.pi / 3.0:
            v_target = 0.0
            w_target = float(np.clip(2.0 * heading_error, -self.max_angular_v, self.max_angular_v))

        # 4. Apply Kinematic Acceleration Clipping
        max_dv = self.max_linear_a * dt
        max_dw = self.max_angular_a * dt

        v = float(np.clip(v_target, self.current_v - max_dv, self.current_v + max_dv))
        w = float(np.clip(w_target, self.current_w - max_dw, self.current_w + max_dw))

        # 5. Apply Soft Collision Override
        if c_stop:
            v = 0.0
            # Note: We bypass the acceleration limit here so it brakes hard, 
            # but cap the spin rate to the safe recovery speed.
            w = float(np.clip(w_target, -self.recovery_spin_v, self.recovery_spin_v))

        # 6. Update State and Return
        self.current_v = v
        self.current_w = w

        return v, w, False


    def _select_lookahead(self, pose : np.ndarray, path : List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Pick the farthest point along the path still within the lookahead radius.
        """
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
