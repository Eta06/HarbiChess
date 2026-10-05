"""Known complete canonical trajectories; selected own-search quiet label only."""

from model import features, quiet


def prepare_ordering(path, config, protected):
    from journal_v3 import digest, read, replay
    from learner import prepare

    # Original exact fullhistory/RNG/prior/terminal replay and group admission.
    _, _, base_receipt, base_sha = prepare(path, config, protected)
    state = read(path)
    if state["actions"] != 16384 or config["max_actions"] != 16384:
        raise ValueError("fixedFINAL16384 only")
    packets = replay(state, config)
    unique, train, val = set(), {}, {}
    import chess

    count = dict(
        quiet_rows=0, excluded_nonquiet_or_explored=0, excluded_protected_games=0, duplicate_games=0
    )
    for _, index, positions, _ in packets:
        game = state["games"][index]
        root = config["roots"][game["root_index"]]
        texts = [r["action"] for r in game["moves"]]
        trajectory = digest(dict(root_fen=root["root_fen"], prefix=root["prefix"], moves=texts))
        board = chess.Board(root["root_fen"])
        touched = {" ".join(board.fen().split()[:4])}
        for text in [*root["prefix"], *texts]:
            board.push_uci(text)
            touched.add(" ".join(board.fen().split()[:4]))
        if touched & set(protected):
            count["excluded_protected_games"] += 1
            continue
        if trajectory in unique:
            count["duplicate_games"] += 1
            continue
        unique.add(trajectory)
        rows = []
        for action_index, ((board, _, _), row) in enumerate(
            zip(positions, game["moves"], strict=True)
        ):
            selected = chess.Move.from_uci(row["selected"])
            # Exploration rows excluded even if random action happened to equal selected.
            if row["explored"] or row["action"] != row["selected"] or not quiet(board, selected):
                count["excluded_nonquiet_or_explored"] += 1
                continue
            choices = sorted(
                [m for m in board.legal_moves if quiet(board, m)], key=lambda m: m.uci()
            )
            if len(choices) < 2:
                continue
            packets = [features(board, move) for move in choices]
            rows.append(
                dict(
                    features=packets,
                    target=choices.index(selected),
                    actions=[m.uci() for m in choices],
                    action_index=action_index,
                )
            )
            count["quiet_rows"] += 1
        if rows:
            (val if int(trajectory[-2:], 16) % 5 == 0 else train)[trajectory] = rows
    if len(train) + len(val) < 16 or count["quiet_rows"] < 1024 or not train or not val:
        raise ValueError("16 distinct quiet trajectories/1024rows/nonempty fixedsplit required")
    receipt = dict(
        schema="own-search-quiet-ordering-data-v1",
        base_receipt=base_receipt,
        base_dataset_sha256=base_sha,
        **count,
        training_rows=sum(map(len, train.values())),
        training_games=len(train),
        validation_rows=sum(map(len, val.values())),
        validation_games=len(val),
        sampling="game-uniform-then-row-uniform;quiet-CE-all-legal-quiet",
        scope="endogenous own-search-imitation;not optimal-move labels",
    )
    return train, val, receipt, digest(dict(train=train, val=val, receipt=receipt))
