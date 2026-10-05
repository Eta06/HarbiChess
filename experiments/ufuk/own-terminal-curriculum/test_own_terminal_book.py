import gzip
import json
import random

import chess
import pytest
from harbichess.training.torch_online_learner import read_online_train_book

from own_terminal_book import DISTANCES, build, completed_games


def fixture(tmp_path):
    # A deterministic own legal game; stop at its actual rules terminal, no engine labels.
    board = chess.Board()
    rng = random.Random(11)
    rows = []
    while board.outcome(claim_draw=True) is None and len(rows) < 1000:
        pre = {'root_fen': chess.STARTING_FEN, 'moves': [m.uci() for m in board.move_stack]}
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        outcome = board.outcome(claim_draw=True)
        rows.append({'game_index': 0, 'source_id': 'TRAIN-own', 'pre': pre,
                     'action': move.uci(), 'post': {'root_fen': chess.STARTING_FEN,
                     'moves': [m.uci() for m in board.move_stack]}, 'rollout_cutoff': False,
                     'terminal_result': None if outcome is None else outcome.result()})
    assert board.outcome(claim_draw=True) is not None and len(rows) >= 32
    original = tmp_path / 'book.json'
    original.write_text(json.dumps({'schema': 1, 'splits': {'train': [
        {'source_game': 'TRAIN-own', 'opening': {'root_fen': chess.STARTING_FEN, 'moves': []}}]}}))
    journal = tmp_path / 'epoch-00000001.json.gz'
    with gzip.open(journal, 'wt') as stream:
        stream.write(json.dumps({'collection': {'actions': [{'transition': x} for x in rows]}},
                                separators=(',', ':')))
    return original, journal, rows


def test_exact_mix_legal_histories_loader_and_prior_label_not_in_opening(tmp_path):
    original, journal, _ = fixture(tmp_path)
    packet, games = build(original, [journal], seed=29)
    assert len(packet['splits']['train']) == 4 and len(games) == 1
    destination = tmp_path / 'mixed.json'
    destination.write_text(json.dumps(packet))
    openings = read_online_train_book(destination)
    assert len({x.source_id for x in openings}) == 4
    assert all(x.state.ply == 0 for x in openings[:3])
    own = packet['splits']['train'][-1]
    distance = own['curriculum_provenance']['distance']
    assert distance in DISTANCES
    assert openings[-1].state.moves == games[0]['terminal'].moves[:-distance]
    assert 'terminal_result' not in own['opening']
    assert packet == build(original, [journal], seed=29)[0]


def test_unknown_cutoff_is_not_completed_and_illegal_or_wrong_result_rejects(tmp_path):
    original, journal, rows = fixture(tmp_path)
    rows[-1]['terminal_result'] = None
    rows[-1]['rollout_cutoff'] = True
    def save():
        with gzip.open(journal, 'wt') as stream:
            stream.write(json.dumps({'collection': {'actions': [{'transition': x} for x in rows]}},
                                    separators=(',', ':')))
    save()
    with pytest.raises(ValueError, match='no own completed'):
        completed_games([journal], original)
    actual = chess.Board()
    for uci in rows[-1]['post']['moves']:
        actual.push_uci(uci)
    rows[-1]['terminal_result'] = (
        '0-1' if actual.result(claim_draw=True) != '0-1' else '1-0')
    save()
    with pytest.raises(ValueError, match='terminal marker'):
        completed_games([journal], original)
    rows[0]['post']['moves'] = ['e2e5']
    save()
    with pytest.raises(ValueError, match='post state'):
        completed_games([journal], original)


def test_fresh_continuation_never_uses_observed_source_terminal_result(tmp_path):
    from harbichess.chess.actions import legal_action_indices, move_to_action
    from harbichess.core.state import ChessMove
    from harbichess.selfplay.online_actor import ActorOpening, OnlineActorConfig, OnlineActors

    original, journal, _ = fixture(tmp_path)
    packet, games = build(original, [journal], seed=29)
    destination = tmp_path / 'mixed.json'
    destination.write_text(json.dumps(packet))
    root = read_online_train_book(destination)[-1].state
    actor = OnlineActors((ActorOpening('own-fresh-test', root),),
                         config=OnlineActorConfig(1, 1, True, 1.0), rng=random.Random(99))
    board = actor.rules.board(root)
    legal = legal_action_indices(board)
    # A source-terminal game is used only to choose a start. Pick a new legal
    # continuation that stays nonterminal, then cap it UNKNOWN after one ply.
    move = next(m for m in board.legal_moves if
                actor.rules.outcome(actor.rules.apply(root, ChessMove(m.uci())),
                    claim_draw=True) is None)
    selected = legal.index(move_to_action(board, move))
    transition = actor.step((tuple(float(i == selected) for i in range(len(legal))),))[0]
    assert games[0]['observed_result_provenance_only'] is not None
    assert transition.terminal_result is None and transition.rollout_cutoff
