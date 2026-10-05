"""Method8 chronological audit; exact mate receipts are independently rules-replayed."""

import gzip
import hashlib
import json
import math
import random
from collections import defaultdict

from own8_audit_support import guard, sha


def replay_mate_certificate_inventory(state, listed, rules, *, claim_draw):
    """Independently replay every legal one-ply mate from the full state history."""
    from harbichess.core.state import ChessMove, TerminalResult

    assert type(listed) is list and listed == sorted(set(listed))
    assert all(type(move) is str for move in listed)
    if rules.outcome(state, claim_draw=claim_draw) is not None:
        assert listed == []
        return ()
    board = rules.inspect(state).copy(stack=True)
    mover_is_white = board.turn
    exact = []
    for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
        if not board.gives_check(move):
            continue
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if not mate:
            continue
        post = rules.apply(state, ChessMove(move.uci()))
        outcome = rules.outcome(post, claim_draw=claim_draw)
        assert outcome is not None and outcome.termination == "checkmate"
        assert outcome.result != TerminalResult.DRAW
        assert (outcome.result == TerminalResult.WHITE_WIN) == mover_is_white
        exact.append(move.uci())
    assert listed == exact
    return tuple(exact)


def audit_winning_root_receipt(
    state, receipt, legal_actions, rules, *, claim_draw, simulations
):
    from harbichess.chess.actions import move_to_action

    exact = replay_mate_certificate_inventory(
        state, receipt.get("certified_mates"), rules, claim_draw=claim_draw
    )
    board = rules.inspect(state)
    by_move = {row["move"]: row for row in receipt["moves"]}
    legal_moves = {
        move.uci(): move_to_action(board, move) for move in board.legal_moves
    }
    assert len(by_move) == len(receipt["moves"]) and set(by_move) == set(legal_moves)
    assert all(legal_moves[move] in legal_actions for move in by_move)
    assert all(
        type(item["visits"]) is int and item["visits"] >= 0 for item in receipt["moves"]
    )
    assert receipt["simulations"] == simulations
    assert sum(item["visits"] for item in receipt["moves"]) == simulations
    assert len(receipt["search_policy"]) == len(legal_actions)
    assert all(math.isfinite(p) and p >= 0 for p in receipt["search_policy"])
    assert math.isclose(math.fsum(receipt["search_policy"]), 1.0, abs_tol=1e-9)
    if exact:
        assert all(p > 0 for p in receipt["search_policy"])
        selected = receipt["selected_action"]
        assert selected in exact and by_move[selected]["visits"] == simulations
        assert receipt["root_value"] == 1.0
        action_to_move = {action: move for move, action in legal_moves.items()}
        nonmate_mass = math.fsum(
            probability
            for action, probability in zip(
                legal_actions, receipt["search_policy"], strict=True
            )
            if action_to_move[action] not in exact
        )
        assert nonmate_mass <= 1e-9
    return len(exact)


def audit_certified_root_receipt(
    state, receipt, legal_actions, rules, *, claim_draw, simulations
):
    """Replay full-history visited losses independently of the producer shield helper."""
    from harbichess.chess.actions import move_to_action
    from harbichess.core.state import ChessMove

    wins = audit_winning_root_receipt(
        state,
        receipt,
        legal_actions,
        rules,
        claim_draw=claim_draw,
        simulations=simulations,
    )
    board = rules.inspect(state)
    action_moves = {
        move_to_action(board, move): move.uci() for move in board.legal_moves
    }
    ordered_moves = [action_moves[action] for action in legal_actions]
    stats = {item["move"]: item for item in receipt["moves"]}
    checked = (
        []
        if wins
        else sorted(move for move, item in stats.items() if item["visits"] > 0)
    )
    assert receipt["visited_loss_checked_actions"] == checked
    losses = []
    for move in checked:
        child = rules.apply(state, ChessMove(move))
        if rules.outcome(child, claim_draw=claim_draw) is not None:
            continue
        child_board = rules.inspect(child).copy(stack=True)
        replies = []
        for reply in sorted(child_board.legal_moves, key=lambda item: item.uci()):
            child_board.push(reply)
            mate = child_board.is_checkmate()
            child_board.pop()
            if mate:
                post = rules.apply(child, ChessMove(reply.uci()))
                outcome = rules.outcome(post, claim_draw=claim_draw)
                assert outcome is not None and outcome.termination == "checkmate"
                assert outcome.value_for(rules.view(child).side_to_move) == 1
                replies.append(reply.uci())
        if replies:
            losses.append({"move": move, "opponent_mates": replies})
    assert receipt["certified_losing_actions"] == losses
    raw = receipt["raw_search_policy"]
    assert len(raw) == len(legal_actions) and all(
        math.isfinite(p) and p >= 0 for p in raw
    )
    assert math.isclose(math.fsum(raw), 1.0, abs_tol=1e-9, rel_tol=0)
    selected = receipt["raw_selected_action"]
    assert selected in stats
    epsilon = 1e-12
    assert receipt["loss_shield_epsilon"] == epsilon
    losing = {item["move"] for item in losses}
    expected = list(raw)
    if wins:
        status = "winning-certificate-precedence"
    elif not losing:
        status = "no-visited-loss-certificate"
    elif len(losing) == len(ordered_moves):
        status = "all-legal-actions-certified-loss-no-safe-action"
    else:
        status = "visited-losses-epsilon-shielded"
        candidates = [move for move in ordered_moves if move not in losing]
        total = math.fsum(
            max(p, epsilon)
            for move, p in zip(ordered_moves, raw, strict=True)
            if move not in losing
        )
        remaining = 1.0 - epsilon * len(losing)
        expected = [
            epsilon if move in losing else remaining * max(p, epsilon) / total
            for move, p in zip(ordered_moves, raw, strict=True)
        ]
        if selected in losing:
            selected = min(
                candidates,
                key=lambda move: (
                    -stats[move]["visits"],
                    -stats[move]["mean_value"],
                    -stats[move]["prior"],
                    move,
                ),
            )
    total = math.fsum(expected)
    expected = [p / total for p in expected]
    assert (
        receipt["loss_shield_status"] == status
        and receipt["selected_action"] == selected
    )
    reason = "raw-search-selection"
    if selected == receipt["raw_selected_action"] and selected in losing:
        reason = "original-certified-losing-action-no-safe-alternative"
    elif selected != receipt["raw_selected_action"]:
        reason = (
            "fallback-visited-nonloss"
            if stats[selected]["visits"] > 0
            else "fallback-unvisited-unknown"
        )
    assert receipt["selected_action_reason"] == reason
    assert all(
        math.isclose(actual, target, abs_tol=1e-15, rel_tol=0)
        for actual, target in zip(receipt["search_policy"], expected, strict=True)
    )
    return wins


def check_chain(record, parent, expected):
    value = dict(record)
    digest = value.pop("sample_chain_sha256")
    assert value["previous_sample_chain_sha256"] == parent
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert (
        hashlib.sha256(bytes.fromhex(parent) + payload).hexdigest()
        == digest
        == expected
    )


def prescribe_roots(epoch, candidates, rules):
    by_mover = {True: [], False: []}
    seen = set()
    for index in candidates:
        row = epoch.actions[index]
        if row.transition.pre in seen:
            continue
        seen.add(row.transition.pre)
        by_mover[rules.view(row.transition.pre).side_to_move.value == "white"].append(
            index
        )
    assert all(len(rows) >= 9 for rows in by_mover.values()), (
        "Fixed FIRST8+LAST8 chronology cannot cover9eachmover; no adaptiveK"
    )
    selected = []
    for rows in by_mover.values():
        rows.sort(key=lambda i: (epoch.actions[i].transition.pre.ply, i))
        selected += [rows[round(i * (len(rows) - 1) / 8)] for i in range(9)]
    assert len(set(selected)) == 18
    return sorted(selected)


def compare_terminal_packet(actual, expected):
    assert actual == expected


def compare_search_packet(actual, expected, canonical):
    assert canonical(actual) == canonical(expected)


def compare_frozen_storage(actual, expected, bits_equal):
    assert bits_equal(actual, expected)


def actor_packet_groups(prescribed, count, actors=128):
    assert count % actors == 0 and len(set(prescribed)) == len(prescribed)
    assert all(type(index) is int and 0 <= index < count for index in prescribed)
    return [
        list(range(start, start + actors))
        for start in sorted({index // actors * actors for index in prescribed})
    ]


def compare_actor_packet(row, policy, base_policy, wdl, base_wdl):
    def normalized(values):
        assert all(math.isfinite(x) and x >= 0 for x in values)
        total = math.fsum(values)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        return tuple(x / total for x in values)

    policy, base_policy = normalized(policy), normalized(base_policy)
    assert policy == row.policy and base_policy == row.base_policy
    assert (
        normalized(wdl) == row.online_pre_wdl and normalized(base_wdl) == row.base_wdl
    )
    # Actual mixed search/raw mu is independently checked for ALL32768 rows during replay.


def choose_chronological_groups(groups, steps):
    assert steps >= 16
    starts = [g["first_collection_index"] for g in groups]
    assert starts == sorted(set(starts))
    assert all(
        type(start) is int and start % 128 == 0 and 0 <= start < steps * 128
        for start in starts
    )
    assert all(g["actor_batch_size"] == 128 for g in groups)
    return {
        i
        for i, group in enumerate(groups)
        if group["first_collection_index"] // 128 < 8
        or group["first_collection_index"] // 128 >= steps - 8
    }


def audit_epoch(a, config, paths, initial, index, neural):
    import torch
    from own8_adapter_controls import JOURNAL_SCHEMA, NATIVE_SCHEMA, TRAINING_SCHEMA

    from harbichess.chess.actions import legal_action_indices
    from harbichess.selfplay.online_epoch import _tuplify
    from harbichess.training.fullgame_own_targets import (
        GameBalancedSampler,
        build_fullgame_targets,
    )
    from harbichess.training.search_acting_epoch import deserialize_search_acting_epoch
    from harbichess.training.torch_search_acting_learner import (
        TorchSearchActingLearner,
        canonical,
        tensor_bits_equal,
    )
    from harbichess.training.torch_search_acting_run import clean_source

    cp = a.run / "checkpoints"
    artifacts = [
        f for n in (index - 1, index) for f in (cp / f"epoch-{n:08d}").iterdir()
    ]
    journal = a.run / "journal" / f"epoch-{index:08d}.json.gz"
    artifacts.append(journal)
    before = {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    before_inputs = {k: sha(v) for k, v in paths.items()}
    initial_models = {
        name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for name, model in [("online", initial.online), ("base", initial.base)]
    }
    schedule_before = initial.schedule_rng.getstate()
    search_before = [r.getstate() for r in initial.search_rngs]
    sampler_seed = initial.sampler_seed_rng.randrange(2**63)
    sampler_after = initial.sampler_seed_rng.getstate()
    final = TorchSearchActingLearner.resume(
        cp / f"epoch-{index:08d}",
        config=config,
        input_paths=paths,
        source_commit=a.source_commit,
    )
    guard(a.deadline_epoch, a.run)
    assert final.config.device == "cuda:0" and final.closed and final.epoch == index
    assert initial.epoch == index - 1 and initial.closed
    assert final.last_epoch_gzip == journal.read_bytes()
    record = json.loads(gzip.decompress(journal.read_bytes()))
    epoch = deserialize_search_acting_epoch(canonical(record["collection"]))
    assert record["schema"] == JOURNAL_SCHEMA and record["epoch"] == index
    for native_epoch in (index - 1, index):
        native_manifest = json.loads(
            (
                a.run / "checkpoints" / f"epoch-{native_epoch:08d}" / "checkpoint.json"
            ).read_text()
        )
        assert native_manifest["schema"] == NATIVE_SCHEMA
    assert len(epoch.actions) == record["fresh_transitions"] == 32768
    assert final.actors.steps == record["actor_steps"] == 256 * index
    assert record["total_fresh_transitions"] == 32768 * index
    assert (
        record["sampler_seed"] == sampler_seed
        and final.sampler_seed_rng.getstate() == sampler_after
    )
    ledger = record["own_search"]
    assert _tuplify(ledger["schedule_rng_before"]) == schedule_before
    assert [_tuplify(r) for r in ledger["search_rng_before"]] == search_before
    assert _tuplify(ledger["actor_rng_before"]) == initial.actors.rng.getstate()
    assert ledger["actor_cursor_before"] == initial.actors.cursor()
    schedule_rng = random.Random()
    schedule_rng.setstate(schedule_before)
    schedule_pending = {}
    seen_offsets, expected_selected = [], []
    by_index = {root["collection_index"]: root for root in ledger["roots"]}
    # Independently consume actual mixed behavior through the original actor RNG.
    for offset in range(0, len(epoch.actions), config.actors.games):
        guard(a.deadline_epoch, a.run)
        rows = epoch.actions[offset : offset + config.actors.games]
        assert [row.transition.slot for row in rows] == list(range(128))
        active = set()
        for row in rows:
            tr = row.transition
            opening = initial.actors.openings[
                initial.actors.games[tr.slot].opening_index
            ]
            block, block_offset = divmod(
                tr.pre.ply - opening.state.ply, config.search.block_plies
            )
            key = (tr.source_id, tr.game_index, tr.slot, block)
            active.add(key)
            if key not in schedule_pending:
                schedule_pending[key] = schedule_rng.randrange(
                    config.search.block_plies
                )
                seen_offsets.append(
                    dict(
                        source_id=key[0],
                        game_index=key[1],
                        slot=key[2],
                        block=block,
                        chosen_offset=schedule_pending[key],
                        first_collection_index=offset + tr.slot,
                    )
                )
            if block_offset == schedule_pending[key]:
                expected_selected.append(offset + tr.slot)
            from harbichess.training.search_acting_epoch import actor_mu

            receipt = by_index.get(offset + tr.slot)
            actor_policy = tuple(receipt["search_policy"]) if receipt else row.policy
            assert actor_mu(actor_policy) == row.behavior_policy
            assert (
                tuple(legal_action_indices(initial.actors.rules.inspect(tr.pre)))
                == row.legal_actions
            )
        policies = tuple(
            tuple(by_index[offset + slot]["search_policy"])
            if offset + slot in by_index
            else row.policy
            for slot, row in enumerate(rows)
        )
        actual = initial.actors.step(policies)
        from dataclasses import replace

        from harbichess.chess.actions import move_to_action

        for observed, row in zip(actual, rows, strict=True):
            board = initial.actors.rules.inspect(observed.pre)
            action = row.legal_actions.index(
                move_to_action(board, board.parse_uci(observed.action.uci))
            )
            assert (
                replace(observed, policy_probability=row.policy[action])
                == row.transition
            )
    assert (
        ledger["selected_indices"] == expected_selected
        and ledger["offsets"] == seen_offsets
    )
    assert schedule_rng.getstate() == _tuplify(ledger["schedule_rng_after"])
    assert len(ledger["roots"]) == len(expected_selected)
    from harbichess.training.ownsearch_targets import validate_search_ledger

    assert ledger["schema"] == "pre-action-masked-search-behavior-v4"
    assert (
        ledger["behavior"]
        == "raw-T1-except-preselected-search-policy-with-visited-mate1-loss-shield-T1;no-PPO"
    )
    validate_search_ledger(
        epoch,
        dict(ledger, schema="ownsearch-random-block-targets-v2"),
        config.search,
        rules=final.actors.rules,
        claim_draw=config.actors.claim_draw,
    )
    certificate_roots = certificate_moves = loss_roots = loss_actions = 0
    for root_collection_index, receipt in by_index.items():
        count = audit_certified_root_receipt(
            epoch.actions[root_collection_index].transition.pre,
            receipt,
            epoch.actions[root_collection_index].legal_actions,
            final.actors.rules,
            claim_draw=config.actors.claim_draw,
            simulations=config.search.simulations,
        )
        certificate_roots += int(count > 0)
        certificate_moves += count
        loss_roots += int(bool(receipt["certified_losing_actions"]))
        loss_actions += len(receipt["certified_losing_actions"])
    assert canonical(initial.actors.close_policy_epoch()) == canonical(
        epoch.truncations
    )
    assert initial.actors.cursor() == final.actors.cursor() == epoch.next_actor_cursor
    assert (
        initial.actors.rng.getstate()
        == final.actors.rng.getstate()
        == epoch.next_actor_rng_state
    )
    # Terminal labels independently derived from final exact-rules outcomes and each mover.
    groups = defaultdict(list)
    for row in epoch.actions:
        groups[(row.transition.source_id, row.transition.game_index)].append(row)
    expected = {}
    unknown = 0
    for rows in groups.values():
        guard(a.deadline_epoch, a.run)
        outcome = final.actors.rules.outcome(
            rows[-1].transition.post, claim_draw=config.actors.claim_draw
        )
        if outcome is None:
            unknown += len(rows)
            continue
        for row in rows:
            tr = row.transition
            value = outcome.value_for(final.actors.rules.view(tr.pre).side_to_move)
            expected[tr] = tuple(float(value == v) for v in (1, 0, -1))
    targets = build_fullgame_targets(
        final.actors.rules, epoch, claim_draw=config.actors.claim_draw
    )
    assert (
        len(targets.targets)
        == len(expected)
        == record["training"]["trained_transitions"]
    )
    assert (
        targets.excluded_actions
        == unknown
        == record["target_counts"]["excluded_actions"]
    )
    for target in targets.targets:
        compare_terminal_packet(target.target_wdl, expected[target.transition])
    from dataclasses import fields

    assert record["target_counts"] == {
        field.name: getattr(targets, field.name)
        for field in fields(targets)
        if field.name != "targets"
    }
    known_groups = defaultdict(list)
    for target in targets.targets:
        known_groups[(target.source_id, target.game_index)].append(target)
    inverse_game_weights = {
        str(key): 1 / (len(known_groups) * len(rows))
        for key, rows in known_groups.items()
    }
    assert (
        not known_groups
        or abs(
            sum(
                inverse_game_weights[str(key)] * len(rows)
                for key, rows in known_groups.items()
            )
            - 1
        )
        < 1e-12
    )
    # Reproduce accepted sampler draws; rejected passes restore both pass-start RNGs.
    training = record["training"]
    assert training["schema"] == TRAINING_SCHEMA
    assert (
        training["behavior_kl_reference"]
        == "frozen-raw-network-policy-on-searched-roots"
    )
    assert (
        final.optimizer_attempted_updates
        == initial.optimizer_attempted_updates + training["optimizer_steps_attempted"]
    )
    assert (
        final.optimizer_accepted_updates
        == initial.optimizer_accepted_updates + training["optimizer_steps_committed"]
    )
    assert (
        final.optimizer_rejected_updates
        == initial.optimizer_rejected_updates + training["optimizer_steps_discarded"]
    )
    assert (
        training["accepted_passes"] + training["rejected_passes"]
        <= config.schedule.passes
    )
    assert training["rejected_passes"] in (0, 1)
    policy_rng = random.Random(sampler_seed)
    value_sampler = (
        GameBalancedSampler(targets.targets, seed=sampler_seed ^ 0xBA71)
        if targets.targets
        else None
    )
    sampler_initial = dict(
        policy=policy_rng.getstate(),
        value=value_sampler.rng.getstate() if value_sampler else None,
    )
    assert canonical(sampler_initial) == canonical(
        training["sampler_rng_state_before_epoch"]
    )
    assert (
        training["optimizer_steps_committed"]
        == training["accepted_passes"] * training["minibatches_per_pass"]
    )
    assert (
        training["optimizer_steps_attempted"]
        == (training["accepted_passes"] + training["rejected_passes"])
        * training["minibatches_per_pass"]
    )
    for _ in range(training["optimizer_steps_committed"]):
        guard(a.deadline_epoch, a.run)
        policy_rng.choices(ledger["roots"], k=config.schedule.minibatch_size)
        if value_sampler:
            value_sampler.sample(config.schedule.minibatch_size)
    sampler_final = dict(
        policy=policy_rng.getstate(),
        value=value_sampler.rng.getstate() if value_sampler else None,
    )
    assert canonical(sampler_final) == canonical(final.last_sampler_rng_state)
    # Every base/behavior tensor and all unused material storage bits, including signed zero.
    material_elements = material_nonzero = 0
    material_sha = {}
    import hashlib

    for name, value in final.base.state_dict().items():
        compare_frozen_storage(
            value.cpu(), initial_models["base"][name], tensor_bits_equal
        )
        assert tensor_bits_equal(
            final.behavior.state_dict()[name].cpu(), initial_models["online"][name]
        )
        if name.startswith("material_value_linear."):
            for model in (final.online, final.base, final.behavior):
                assert tensor_bits_equal(
                    model.state_dict()[name].cpu(), initial_models["base"][name]
                )
            material_elements += value.numel()
            material_nonzero += int(torch.count_nonzero(value))
            material_sha[name] = hashlib.sha256(
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
    assert material_elements == 21
    import chess

    from harbichess.chess.actions import move_to_action
    from harbichess.search.full_gumbel import FullGumbelConfig
    from harbichess.search.ownsearch_wavefront import (
        BatchedPositionEvaluator,
        WavefrontGumbel,
    )
    from harbichess.training.search_acting_epoch import MaskedRootEvaluator, normalized
    from harbichess.training.torch_array_encoder import (
        TorchArrayBoardEncoder,
        TorchArrayPolicyValueBackend,
    )

    groups = ledger["groups"]
    expected_groups = []
    selected_set = set(ledger["selected_indices"])
    for start in range(0, len(epoch.actions), 128):
        slots = [slot for slot in range(128) if start + slot in selected_set]
        if slots:
            expected_groups.append(
                dict(
                    first_collection_index=start,
                    actor_batch_size=128,
                    selected_slots=slots,
                    selected_indices=[start + s for s in slots],
                )
            )
    assert groups == expected_groups
    total_steps = len(epoch.actions) // 128
    chosen = choose_chronological_groups(groups, total_steps) if neural else set()
    candidates = [i for g in sorted(chosen) for i in groups[g]["selected_indices"]]
    prescribed = (
        prescribe_roots(epoch, candidates, final.actors.rules) if neural else []
    )
    rngs = [random.Random() for _ in search_before]
    for rng, state in zip(rngs, search_before, strict=True):
        rng.setstate(state)
    root_by_index = {root["collection_index"]: root for root in ledger["roots"]}
    evaluator = BatchedPositionEvaluator(
        TorchArrayPolicyValueBackend(final.behavior, device=config.device),
        final.actors.rules,
    )
    evaluator.encoder = TorchArrayBoardEncoder(final.actors.rules)
    audited = 0
    for group_index, group in enumerate(groups):
        guard(a.deadline_epoch, a.run)
        indices = group["selected_indices"]
        if group_index not in chosen:
            for i in indices:
                for _ in epoch.actions[i].legal_actions:
                    rngs[epoch.actions[i].transition.slot].random()
            continue
        start = group["first_collection_index"]
        rows = epoch.actions[start : start + 128]
        assert len(rows) == 128 and [row.transition.slot for row in rows] == list(
            range(128)
        )
        slots = group["selected_slots"]
        assert (
            indices == [start + slot for slot in slots]
            and group["actor_batch_size"] == 128
        )
        states = tuple(row.transition.pre for row in rows)
        masked = MaskedRootEvaluator(
            evaluator, states, [slot in slots for slot in range(128)]
        )
        search = WavefrontGumbel(
            masked,
            final.actors.rules,
            FullGumbelConfig(
                config.search.simulations,
                config.search.max_considered_actions,
                config.search.gumbel_scale,
                config.search.value_scale,
                config.search.maxvisit_init,
                config.actors.claim_draw,
            ),
        )
        results = search.search_many(
            [states[slot] for slot in slots],
            [rngs[slot] for slot in slots],
            guard=lambda: guard(a.deadline_epoch, a.run),
        )
        for slot, i, result in zip(slots, indices, results, strict=True):
            row = rows[slot]
            board = final.actors.rules.inspect(states[slot])
            by_action = {
                move_to_action(board, chess.Move.from_uci(move.uci)): weight
                for move, weight in result.action_weights
            }
            receipt = dict(
                collection_index=i,
                legal_actions=list(row.legal_actions),
                search_policy=list(
                    normalized(tuple(by_action[action] for action in row.legal_actions))
                ),
                selected_action=result.selected_action.uci,
                root_value=result.root_value,
                moves=[
                    dict(
                        move=m.move.uci,
                        visits=m.visits,
                        prior=m.prior,
                        mean_value=m.mean_value,
                    )
                    for m in result.moves
                ],
                simulations=result.simulations,
                certified_mates=[move.uci for move in result.certified_mates],
                root_ply=states[slot].ply,
                mover=final.actors.rules.view(states[slot]).side_to_move.value,
            )
            from harbichess.search.mate_certificates import (
                shield_visited_losses,
                visited_mate_in_one_losses,
            )

            visited = tuple(m.move for m in result.moves if m.visits > 0)
            checked = () if result.certified_mates else visited
            losses = (
                visited_mate_in_one_losses(
                    final.actors.rules,
                    states[slot],
                    checked,
                    claim_draw=config.actors.claim_draw,
                )
                if checked
                else ()
            )
            root_moves = tuple(move for move, _ in result.action_weights)
            shielded, selected_move, shield_status = shield_visited_losses(
                root_moves,
                tuple(probability for _, probability in result.action_weights),
                result.moves,
                tuple(move for move, _ in losses),
                selected_action=result.selected_action,
                certified_wins=result.certified_mates,
                epsilon=1e-12,
            )
            raw_policy = receipt["search_policy"]
            shield_by_action = {
                move_to_action(board, chess.Move.from_uci(move.uci)): weight
                for move, weight in zip(root_moves, shielded, strict=True)
            }
            reason = "raw-search-selection"
            losing = {move for move, _ in losses}
            if selected_move == result.selected_action and selected_move in losing:
                reason = "original-certified-losing-action-no-safe-alternative"
            elif selected_move != result.selected_action:
                stats = {m.move: m for m in result.moves}
                reason = (
                    "fallback-visited-nonloss"
                    if stats[selected_move].visits > 0
                    else "fallback-unvisited-unknown"
                )
            receipt.update(
                raw_search_policy=raw_policy,
                search_policy=list(
                    normalized(
                        tuple(shield_by_action[action] for action in row.legal_actions)
                    )
                ),
                raw_selected_action=result.selected_action.uci,
                selected_action=selected_move.uci,
                selected_action_reason=reason,
                visited_loss_checked_actions=sorted(move.uci for move in checked),
                certified_losing_actions=[
                    {
                        "move": move.uci,
                        "opponent_mates": [reply.uci for reply in replies],
                    }
                    for move, replies in losses
                ],
                loss_shield_status=shield_status,
                loss_shield_epsilon=1e-12,
            )
            compare_search_packet(receipt, root_by_index[i], canonical)
        audited += len(indices)
    assert [rng.getstate() for rng in rngs] == [
        _tuplify(state) for state in ledger["search_rng_after"]
    ]
    from harbichess.training.torch_fullgame_ppo import make_torch_epoch_inference

    infer = make_torch_epoch_inference(
        final.behavior, final.base, final.encoder, device=config.device
    )
    raw_groups = actor_packet_groups(prescribed, len(epoch.actions))
    raw_references = {}
    for indices in raw_groups:
        guard(a.deadline_epoch, a.run)
        rows = [epoch.actions[i] for i in indices]
        output = infer(
            tuple(row.transition.pre for row in rows),
            tuple(row.legal_actions for row in rows),
        )
        for position, i in enumerate(indices):
            if i not in prescribed:
                continue
            row = rows[position]
            raw_references[i] = {
                "policy": output.policy[position],
                "base_policy": output.base_policy[position],
                "wdl": output.wdl[position],
                "base_wdl": output.base_wdl[position],
            }
            # RAW packets match own inference; actualmu comes from search policy at selected roots.

            compare_actor_packet(
                row,
                output.policy[position],
                output.base_policy[position],
                output.wdl[position],
                output.base_wdl[position],
            )
            assert row.behavior_policy == actor_mu(root_by_index[i]["search_policy"])
    neural_positions = sum(evaluator.batch_sizes)
    raw_duplicate_positions = 2 * 128 * len(raw_groups)
    assert before == {str(f.relative_to(a.run)): sha(f) for f in artifacts}
    assert before_inputs == {k: sha(v) for k, v in paths.items()}
    check_chain(record, initial.sample_chain_sha256, final.sample_chain_sha256)
    assert initial.schedule_rng.getstate() == _tuplify(ledger["schedule_rng_before"])
    assert final.schedule_rng.getstate() == _tuplify(ledger["schedule_rng_after"])
    assert [r.getstate() for r in final.search_rngs] == [
        _tuplify(x) for x in ledger["search_rng_after"]
    ]
    assert record["training"]["policy_target_rows"] == len(ledger["roots"])
    assert canonical(training["sampler_rng_state"]) == canonical(sampler_final)
    guard(a.deadline_epoch, a.run)
    clean_source(a.source_commit)
    report = dict(
        epoch=index,
        native_artifact_sha256=before,
        journal_sha256=sha(journal),
        raw_actor_replayed=32768,
        known_value_rows=len(expected),
        unknown_value_rows=unknown,
        search_target_rows=len(ledger["roots"]),
        independently_replayed_mate_certificate_roots=certificate_roots,
        independently_replayed_visited_loss_roots=loss_roots,
        independently_replayed_visited_loss_actions=loss_actions,
        visited_loss_inventory_fullhistory_and_actual_mu_verified=True,
        independently_replayed_mate_certificate_actions=certificate_moves,
        exact_certificate_inventory_and_fullhistory_rules=True,
        certified_root_visit_budget_verified=True,
        certified_root_full_legal_policy_support_verified=True,
        optimizer_attempts=training["optimizer_steps_attempted"],
        optimizer_committed=training["optimizer_steps_committed"],
        sample_chain_sha256=final.sample_chain_sha256,
        schedule_and_all128search_rngs_verified=True,
        sampler_policy_root_uniform_value_game_uniform=True,
        prescribed_neural_roots_verified=len(prescribed),
        prescribed_collection_indices=prescribed,
        exact_fullgroup_neural_roots_recomputed=audited,
        duplicate_neural_positions=neural_positions + raw_duplicate_positions,
        raw_actor_packet_roots_verified=len(prescribed),
        raw_actor_packet_recomputed_reference=raw_references,
        raw_actor_full128_groups=len(raw_groups),
        raw_actor_duplicate_neural_positions_both_models=raw_duplicate_positions,
        neural_witness_K=8,
        audited_chronological_groups=len(chosen),
        neural_selection=(
            "Fixed FIRST8+LAST8 chronological128actor-root batches+selected simulation masks; "
            "18history-dedup roots9eachmover, all selected masks exactcompared"
        ),
        material_elements=material_elements,
        material_nonzero=material_nonzero,
        material_storage_sha256=material_sha,
        source_commit=a.source_commit,
        optimizer_updates_performed_by_audit=0,
    )
    return final, report
