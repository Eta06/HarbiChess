"""Outcome-blind balanced selection; current-state/complete-history exclusion."""

import hashlib


def canonical_key(board):
    return " ".join(board.fen().split()[:4])


def history_sha(root_fen, prefix):
    return hashlib.sha256((root_fen + "\n" + " ".join(prefix)).encode()).hexdigest()


def select(groups, *, seed, count, excluded_positions, excluded_histories):
    if type(seed) is not int or seed < 0 or type(count) is not int or count <= 0:
        raise ValueError("fixed seed/count")
    ranked, seen_ids = {}, set()
    for game, rows in groups.items():
        eligible = []
        for row in rows:
            if row["trajectory_id"] != game or row["row_id"] in seen_ids:
                raise ValueError("unique stable TRAIN trajectory/row identity")
            seen_ids.add(row["row_id"])
            if row["fen4"] in excluded_positions or row["history_sha256"] in excluded_histories:
                continue
            eligible.append(row)
        if eligible:
            ranked[game] = sorted(
                eligible,
                key=lambda r: hashlib.sha256(
                    f"NNUE-teacher4096-v1|{seed}|{game}|{r['row_id']}".encode()
                ).digest(),
            )
    games = sorted(
        ranked, key=lambda g: hashlib.sha256(f"NNUE-teacher-game-v1|{seed}|{g}".encode()).digest()
    )
    cursors = {g: 0 for g in games}
    selected, histories = [], set()
    while len(selected) < count:
        progressed = False
        for game in games:
            while cursors[game] < len(ranked[game]):
                row = ranked[game][cursors[game]]
                cursors[game] += 1
                if row["history_sha256"] in histories:
                    continue
                selected.append(row)
                histories.add(row["history_sha256"])
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"only{len(selected)} unique admissible TRAIN roots; no relaxation")
    return selected


def teacher_target(score, mover):
    import math

    pov = score.pov(mover)
    cp, mate = pov.score(), pov.mate()
    if mate is not None:
        # Mate0 cannot identify winning side on a nonterminal legal input.
        if mate == 0:
            raise ValueError("ambiguous terminal mate0 on nonterminal root")
        return dict(
            cp_mover=None,
            mate_mover=mate,
            target=1.0 if mate > 0 else -1.0,
            target_is_calibrated_WDL=False,
        )
    if type(cp) is not int:
        raise ValueError("exact finite CP or mate score required")
    return dict(
        cp_mover=cp, mate_mover=None, target=math.tanh(cp / 600), target_is_calibrated_WDL=False
    )
