"""Experimental Q-range damping; historical Full Gumbel defaults stay unchanged."""

import math

from harbichess.search.full_gumbel import FullGumbelMCTS, _Node


class QRangeFlooredMCTS(FullGumbelMCTS):
    def __init__(self, *args, range_floor=1e-8, **kwargs):
        if isinstance(range_floor, bool) or not math.isfinite(range_floor) or range_floor < 1e-8:
            raise ValueError("Q range floor must be finite and at least 1e-8")
        super().__init__(*args, **kwargs)
        self.range_floor = range_floor

    def _completed_q(self, node: _Node):
        scores = super()._completed_q(node)
        if self.range_floor == 1e-8:
            return scores
        visited = [child for child in node.children.values() if child.visit_count]
        visits = sum(child.visit_count for child in visited)
        mass = sum(child.prior for child in visited)
        weighted = sum(child.prior * -child.mean_value / mass for child in visited) if mass else 0
        mixed = (node.raw_value + visits * weighted) / (visits + 1)
        completed = [
            -child.mean_value if child.visit_count else mixed for child in node.children.values()
        ]
        spread = max(max(completed) - min(completed), 1e-8)
        damping = spread / max(spread, self.range_floor)
        return {move: score * damping for move, score in scores.items()}
