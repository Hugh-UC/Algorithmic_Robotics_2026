"""
Pose Graph Data Structure for Graph-Based SLAM (Week 11)

PROVIDED COMPLETE — DO NOT MODIFY.

A thin wrapper around two lists (nodes and edges) plus the covariance ->
information conversion that the optimiser expects. The interesting code lives
in slam_node.py (graph construction) and graph_optimizer.py (Gauss-Newton).
"""

import numpy as np
from typing import List, Tuple


class PoseGraph:
    """Pose graph for 2D SLAM: nodes = poses, edges = relative constraints."""

    def __init__(self):
        self.nodes: List[np.ndarray] = []
        self.edges: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

    def add_node(self, pose: np.ndarray) -> int:
        """Append a pose to the graph and return its index."""
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1

    def add_edge(self, from_id: int, to_id: int,
                 measurement: np.ndarray, covariance: np.ndarray):
        """
        Add a constraint between two pose nodes.

        Converts the measurement covariance to an information matrix
        (Omega = Sigma^-1) which weights the constraint in the optimiser.
        """
        cov_reg = covariance + 1e-6 * np.eye(3)
        omega = np.linalg.inv(cov_reg)
        self.edges.append((from_id, to_id, measurement.copy(), omega))

    def get_poses(self) -> List[np.ndarray]:
        return [p.copy() for p in self.nodes]

    def get_num_nodes(self) -> int:
        return len(self.nodes)

    def get_num_edges(self) -> int:
        return len(self.edges)

    def set_pose(self, node_id: int, pose: np.ndarray):
        self.nodes[node_id] = pose.copy()
