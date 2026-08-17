"""Dynamical equivalence via ε-connectivity graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np


def trajectory_distance(gamma_i: np.ndarray, gamma_j: np.ndarray) -> float:
    """
    d_H(γ_i, γ_j) = max_t |x_i(t) - x_j(t)| / min(x_i(t), x_j(t))
    Relative max-distance over the horizon.
    """
    rel_diff = np.abs(gamma_i - gamma_j) / np.maximum(
        np.minimum(gamma_i, gamma_j), 1e-10
    )
    return float(np.max(rel_diff))


def distance_matrix(paths: np.ndarray) -> np.ndarray:
    """Pairwise trajectory distances for all paths."""
    n = len(paths)
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0.0)
    for i in range(n):
        denom = np.maximum(np.minimum(paths[i], np.min(paths, axis=0)), 1e-10)
        rel_diffs = np.abs(paths[i + 1 :] - paths[i]) / denom
        dist[i, i + 1 :] = np.max(rel_diffs, axis=1)
        dist[i + 1 :, i] = dist[i, i + 1 :]
    return dist


def build_epsilon_graph(
    paths: np.ndarray | None = None,
    epsilon: float = 0.08,
    dist_matrix: np.ndarray | None = None,
) -> Dict[int, List[int]]:
    """Build G_{ε,H}: edge (i,j) iff d_H(γ_i, γ_j) < ε."""
    if dist_matrix is None:
        if paths is None:
            raise ValueError("Provide paths or dist_matrix")
        dist_matrix = distance_matrix(paths)

    n = dist_matrix.shape[0]
    adj = (dist_matrix < epsilon) & (dist_matrix > 0)
    graph: Dict[int, List[int]] = defaultdict(list)
    rows, cols = np.where(adj)
    for i, j in zip(rows, cols):
        if i < j:
            graph[i].append(j)
            graph[j].append(i)
    return graph


def find_components(graph: Dict[int, List[int]], n: int) -> List[List[int]]:
    """Connected components = dynamical equivalence classes."""
    visited: set[int] = set()
    components: List[List[int]] = []
    for start in range(n):
        if start in visited:
            continue
        comp: List[int] = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def divergence_analysis(
    components: List[List[int]], n: int
) -> dict[str, int | float]:
    """Classify trajectory pairs as transient (same class) vs persistent (different)."""
    comp_map = {pid: idx for idx, comp in enumerate(components) for pid in comp}
    transient = persistent = 0
    for i in range(n):
        for j in range(i + 1, n):
            if comp_map[i] == comp_map[j]:
                transient += 1
            else:
                persistent += 1
    total = transient + persistent
    return {
        "transient": transient,
        "persistent": persistent,
        "total": total,
        "persistence_rate": persistent / max(total, 1),
    }
