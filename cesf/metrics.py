"""Operational complexity and compression metrics."""

from __future__ import annotations

import math


def operational_complexity_bits(n_classes: int) -> float:
    if n_classes <= 1:
        return 0.0
    return math.log2(n_classes)


def compression_ratio(n_theoretical: int, n_classes: int) -> float:
    return n_theoretical / max(n_classes, 1)


def categorize_risks(
    normed_scores: list[tuple[list[int], float]],
    n_admissible: int,
    event_threshold: float,
    probable_threshold: float = 0.02,
) -> dict[str, int]:
    cats = {"probable": 0, "improbable_consequential": 0, "negligible": 0}
    for comp, score in normed_scores:
        p = len(comp) / n_admissible
        if score >= event_threshold:
            if p >= probable_threshold:
                cats["probable"] += 1
            else:
                cats["improbable_consequential"] += 1
        else:
            cats["negligible"] += 1
    return cats
