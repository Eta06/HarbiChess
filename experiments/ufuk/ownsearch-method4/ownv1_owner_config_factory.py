"""Build exact sidecar owner configs from an already frozen slot4 registration."""
import argparse
import json
import time
from pathlib import Path

from ownv1_audit_support import publish, sha
from ownv1_strength_config import SEEDS, validate_config

SOURCE = "a278bba67bce962cb9294f0d24040e02e9acf1f4"

AUDIT_HELPERS = ('ownv1_audit_support.py', 'ownv1_full_audit.py',
                 'ownv1_fresh_cli_replay.py')


def audit_manifest(registration, inputs, seed, first, paths, helpers):
    assert registration['status'] == 'frozen-before-formal-execution'
    assert registration['qualification_ledger_slot'] == 4
    assert registration['source_commit'] == SOURCE and seed in SEEDS
    assert inputs['protocol']['sha256'] == sha(paths['registration'])
    return {'schema': 'ufuk-ownv1-formal4-audit-manifest-v1',
            'status': registration['status'], 'qualification_ledger_slot': 4,
            'source_commit': SOURCE, 'fixed_epochs': registration['fixed_epochs'],
            'neural_audit_epochs': registration['neural_audit_epochs'],
            'original_training_started_epoch': first,
            'original_training_deadline_epoch': first +
                registration['whole_training_seconds_per_seed'],
            'whole_training_seconds': registration['whole_training_seconds_per_seed'],
            'whole_audit_seconds': registration['whole_audit_seconds_from_originalfirstclock'],
            'producer_checkout': paths['repo'], 'run': paths['run'], 'inputs': inputs,
            'frozen_config': registration['configs'][str(seed)],
            'helper_sha256': {name: sha(Path(helpers) / name) for name in AUDIT_HELPERS}}


def main():
    p = argparse.ArgumentParser()
    for name in ('registration', 'three-input-manifest', 'qualification-config',
                 'paths-config', 'helpers', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=name != 'helpers')
    a = p.parse_args()
    for name in ('registration', 'three_input_manifest', 'qualification_config', 'paths_config'):
        assert sha(getattr(a, name)) == getattr(a, name + '_sha256')
    reg = json.loads(a.registration.read_text())
    original = json.loads(a.three_input_manifest.read_text())
    q = validate_config(json.loads(a.qualification_config.read_text()))
    paths = json.loads(a.paths_config.read_text())
    clock_path = Path(paths["common_original_firstclock_receipt"])
    assert sha(clock_path) == paths["common_original_firstclock_receipt_sha256"]
    clock = json.loads(clock_path.read_text())
    assert clock["schema"] == "own45-common-original-firstclock-v1"
    assert clock["slots"] == [4, 5]
    common_first = clock["original_training_started_epoch"]
    assert type(common_first) in (int, float) and common_first <= time.time()
    assert reg['source_commit'] == q['source_commit'] == SOURCE
    assert reg['fixed_epochs'] == q['fixed_epochs']
    assert reg['three_input_manifest_sha256'] == a.three_input_manifest_sha256
    for name, digest in q['helper_sha256'].items():
        assert sha(a.helpers / name) == digest
    assert list(map(int, paths['seeds'])) == list(SEEDS)
    a.output.mkdir(exist_ok=False, parents=True)
    inputs4 = {'seeds': {}}
    rows = []
    for seed in SEEDS:
        row = paths['seeds'][str(seed)]
        inputs = {**original['seeds'][str(seed)],
                  'protocol': {'path': str(a.registration.resolve()),
                               'sha256': a.registration_sha256}}
        for info in inputs.values():
            assert sha(info['path']) == info['sha256']
        inputs4['seeds'][str(seed)] = inputs
        first = row['original_training_started_epoch']
        assert first == common_first
        assert reg['earliest_training_epoch'] <= first <= time.time()
        assert first + reg['whole_audit_seconds_from_originalfirstclock'] + \
               reg['posttraining_reserve_seconds'] < 1791170400
        full = audit_manifest(reg, inputs, seed, first,
            {'registration': str(a.registration), 'repo': paths['repo'], 'run': row['run']},
            a.helpers)
        manifest_path = a.output / f'audit-manifest-{seed}.json'
        publish(manifest_path, full)
        rows.append({**row, 'seed': seed, 'audit_manifest': str(manifest_path.resolve())})
    publish(a.output / 'four-input-manifest.json', inputs4)
    post = {**paths['posttraining'], 'source_commit': SOURCE, 'repo': paths['repo'],
            'python': paths['python'], 'helpers': str(a.helpers.resolve()),
            'qualification_config': str(a.qualification_config.resolve()),
            'qualification_config_sha256': a.qualification_config_sha256,
            'whole_training_seconds': reg['whole_training_seconds_per_seed'],
            'whole_audit_seconds': reg['whole_audit_seconds_from_originalfirstclock'],
            'seeds': rows}
    publish(a.output / 'posttraining-config.json', post)


if __name__ == '__main__':
    main()
