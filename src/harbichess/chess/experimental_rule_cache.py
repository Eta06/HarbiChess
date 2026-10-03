"""Retained rejected experiment: immutable rule facts, never enabled by default."""

from collections import OrderedDict

from harbichess.chess.rules import PythonChessRules


class CachedPythonChessRules(PythonChessRules):
    """Correct histories, but AYNA's 16-simulation speed gate did not pass."""

    def _facts(self, state) -> dict:
        cache = getattr(self._thread_local, "fact_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._thread_local.fact_cache = cache
        if state not in cache:
            cache[state] = {}
        cache.move_to_end(state)
        while len(cache) > self.board_cache_size:
            cache.popitem(last=False)
        return cache[state]

    def view(self, state):
        facts = self._facts(state)
        if "view" not in facts:
            facts["view"] = super().view(state)
        return facts["view"]

    def legal_moves(self, state):
        facts = self._facts(state)
        if "legal" not in facts:
            facts["legal"] = super().legal_moves(state)
        return facts["legal"]

    def outcome(self, state, *, claim_draw=False):
        facts = self._facts(state)
        key = "claimed_outcome" if claim_draw else "forced_outcome"
        if key not in facts:
            facts[key] = super().outcome(state, claim_draw=claim_draw)
        return facts[key]
