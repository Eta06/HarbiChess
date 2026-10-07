"""PREdraw deterministic conservative placement-alias ownership across ECO strata."""

import hashlib

import chess


def state_alias(row):
    board = chess.Board(row['root_fen'])
    key = min(board.board_fen(), board.mirror().board_fen())
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little', signed=True)


def own_aliases(groups):
    """Lexicographically first ECO owns each alias BEFORE any random choice.

    Keep every source game within its owning stratum, including same-alias source
    games. Different drawn strata can never share an alias. This changes the
    frozen sampling population prospectively, not the result of any random draw.
    """
    owners = {}
    for eco in sorted(groups):
        for row in groups[eco]:
            owners.setdefault(state_alias(row), eco)
    result = {eco: [r for r in rows if owners[state_alias(r)] == eco]
              for eco, rows in groups.items()}
    return {eco: rows for eco, rows in result.items() if rows}, owners
