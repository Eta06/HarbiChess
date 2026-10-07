"""Original source semantics and typed played-depth metadata, no search calls."""
from pathlib import Path

from audit_known160_v2 import played_depth_valid


def test_played_depth_is_typed_one_to_eight_not_terminal_zero():
    assert all(played_depth_valid(x) for x in range(1, 9))
    assert not any(played_depth_valid(x) for x in (0, -1, 9, True, 1.0, None))


def test_exact_original_source_initialization_and_terminal_only_zero():
    source = Path(__file__).with_name("search.py").read_text()
    assert "return Result(None, terminal, self.nodes, 0, 0, 0)" in source
    assert "best, completed = scores[winner], 1" in source
    assert "for depth in range(1, self.max_depth + 1):" in source
