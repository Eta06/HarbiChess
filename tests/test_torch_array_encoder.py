import os

import chess
import numpy as np
import pytest
import torch
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.training.torch_array_encoder import (
    TorchArrayBoardEncoder as ArrayBoardEncoder,
)

CASES = [
    (chess.STARTING_FEN, 'e2e4 a7a6 e4e5 d7d5 e5d6'),
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1 e8c8'),
    ('7k/P7/8/8/8/8/8/7K w - - 0 1', 'a7a8n'),
    ('7k/8/8/8/8/8/8/R6K w - - 99 1', 'a1a2 h8g8'),
    (chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8'),
]


@pytest.mark.parametrize('fen,moves', CASES)
def test_all104_float32_storage_both_perspectives_history_and_rules(fen, moves):
    rules = PythonChessRules()
    old, new = BoardEncoder(rules), ArrayBoardEncoder(rules)
    state = rules.initial_state(fen)
    states = [state]
    for uci in moves.split():
        states.append(rules.apply(states[-1], ChessMove(uci)))
    for state in states:
        reference, candidate = old.encode(state), new.encode(state)
        expected = np.asarray(reference.values, dtype=np.float32)
        assert expected.tobytes() == candidate.values.tobytes()
        assert candidate.shape == reference.shape and candidate.schema_version == 1
        with pytest.raises(ValueError):
            candidate.values.setflags(write=True)
        with pytest.raises(ValueError):
            candidate.values[0] = 3.0
        unchanged = np.asarray(old.encode(state).values, dtype=np.float32)
        assert unchanged.tobytes() == expected.tobytes()


def test_real_e8_masked_cpu_outputs_storage_and_rng_unchanged():
    import copy
    from pathlib import Path

    import torch
    from harbichess.backends.torch_network import load_weights
    from harbichess.chess.actions import legal_action_indices
    from harbichess.training.torch_fullgame_ppo import make_torch_epoch_inference

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        weights = Path(os.environ.get(
            'HARBICHESS_TEST_E8',
            '/workspace/HarbiChess/artifacts/ufuk-a100-mirror-20261004/'
            'harbichess-inputs/initial-e8.safetensors'))
        network = load_weights(weights)
        base = copy.deepcopy(network)
        before = {k: v.detach().reshape(-1).view(torch.uint8).clone()
                  for k, v in network.state_dict().items()}
        rules = PythonChessRules()
        white = rules.initial_state()
        black = rules.apply(white, ChessMove('e2e4'))
        states = (white, black)
        legal = tuple(legal_action_indices(rules.inspect(s)) for s in states)
        rng_before = torch.get_rng_state().clone()
        old = make_torch_epoch_inference(network, base, BoardEncoder(rules), device='cpu')
        new = make_torch_epoch_inference(network, base, ArrayBoardEncoder(rules), device='cpu')
        assert old(states, legal) == new(states, legal)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert all(torch.equal(before[k], v.detach().reshape(-1).view(torch.uint8))
                   for k, v in network.state_dict().items())
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(),
        reason='Actual CUDA unavailable locally; root must run this branch on A100'))])
def test_private_array_learner_counterfactual_and_fresh_resume_all_payloads(tmp_path, device):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from test_search_acting_stage import PROCESS, SOURCE, WEIGHTS, fixture

    config_path, _ = fixture(tmp_path)
    import json
    data = json.loads(config_path.read_text())
    data['device'] = device
    config_path.write_text(json.dumps(data) + '\n')
    candidate = Path(__file__).resolve().parent.parent / 'src'
    baseline = Path(os.environ.get(
        'HARBICHESS_BASELINE_SRC',
        '/workspace/work/harbichess/own-terminal-curriculum-stage/source-4515/src'))
    assert baseline.is_dir(), 'Pinned4515 isolated baseline required for local witness'
    import hashlib
    pinned_baseline = {
        'training/search_acting_epoch.py':
            '71bc9e86873d3e0f0abc26b0a3ba4d802828b0be9d95486c7a5c59b0660ec339',
        'training/torch_search_acting_learner.py':
            '04b876c56e3ba5dc02a195237138fe49c1af812e6d107580097b89b170de4bc3',
        'training/torch_search_acting_checkpoint.py':
            '00145c9aa1625c6823bdff6c21b285b0907ed3228c799f0ba564e193bafa3426',
    }
    for name, expected in pinned_baseline.items():
        assert hashlib.sha256((baseline / 'harbichess' / name).read_bytes()).hexdigest() == expected
    assert baseline.resolve() != candidate.resolve()

    jobs = [
        (baseline, tmp_path / 'baseline', 'whole'),
        (candidate, tmp_path / 'candidate', 'whole'),
        (candidate, tmp_path / 'split', 'pause'),
        (candidate, tmp_path / 'split', 'resume'),
        (candidate, tmp_path / 'candidate', 'strictload'),
        (candidate, tmp_path / 'split', 'strictload'),
    ]
    worker = PROCESS.replace(
        'root.mkdir(exist_ok=True)',
        "root.mkdir(exist_ok=True)\n"
        "expected=('TorchArrayBoardEncoder' if "
        "__import__('os').environ.get('ARRAY_WITNESS') else 'BoardEncoder')\n"
        "assert type(l.encoder).__name__==expected\n"
        "if l.epoch==0: l.checkpoint(root/'native0')"
    )
    for source_path, output, mode in jobs:
        env = dict(os.environ, PYTHONPATH=str(source_path), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if source_path == candidate:
            env['ARRAY_WITNESS'] = '1'
        else:
            env.pop('ARRAY_WITNESS', None)
        result = subprocess.run([sys.executable, '-c', worker, str(output),
                                 str(config_path), str(WEIGHTS), SOURCE, mode],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
    # SOURCE is explicitly a synthetic unit marker, not an actual clean producer.
    # Each branch starts freshly at e8; no old native source is relabeled.
    for epoch in (0, 1, 2):
        for name in ('model.safetensors', 'base.safetensors', 'behavior.safetensors',
                     'training.pt', 'actor.json', 'last-frozen-epoch.json.gz'):
            expected = (tmp_path / f'baseline/native{epoch}' / name).read_bytes()
            assert (tmp_path / f'candidate/native{epoch}' / name).read_bytes() == expected
            assert (tmp_path / f'split/native{epoch}' / name).read_bytes() == expected
        if epoch == 0:
            continue
        expected = (tmp_path / f'baseline/journal{epoch}.gz').read_bytes()
        assert (tmp_path / f'candidate/journal{epoch}.gz').read_bytes() == expected
        assert (tmp_path / f'split/journal{epoch}.gz').read_bytes() == expected
