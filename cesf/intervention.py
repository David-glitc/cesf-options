"""Causal intervention analysis (Section 5.2)."""

from __future__ import annotations

from cesf.equivalence import build_epsilon_graph, find_components


def causal_significance(
    baseline_paths,
    intervened_paths,
    base_components: list[list[int]],
    int_components: list[list[int]] | None = None,
    epsilon: float | None = None,
) -> dict[str, int | float]:
    if int_components is None:
        if epsilon is None:
            raise ValueError("Provide int_components or epsilon")
        int_graph = build_epsilon_graph(intervened_paths, epsilon)
        int_components = find_components(int_graph, len(intervened_paths))

    base_map = {pid: idx for idx, comp in enumerate(base_components) for pid in comp}
    int_map = {pid: idx for idx, comp in enumerate(int_components) for pid in comp}
    significant = 0
    n = min(len(baseline_paths), len(intervened_paths))
    for pid in range(n):
        if base_map.get(pid) != int_map.get(pid):
            significant += 1
    return {"significant": significant, "total": n, "rate": significant / max(n, 1)}
