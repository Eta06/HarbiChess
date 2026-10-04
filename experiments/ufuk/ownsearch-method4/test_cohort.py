import hashlib
import json
from pathlib import Path

import own45_cohort as cohort
import pytest


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f)


def wait(p, deadline):
    return json.loads(Path(p).read_text())


def fixture(tmp_path):
    members = []
    for slot in (4, 5):
        root = tmp_path / str(slot)
        root.mkdir()
        members.append({'slot': slot, 'source_commit': cohort.SOURCES[slot],
            'root': str(root), 'coordinator_sha256': str(slot) * 64,
            'required_process_receipt_names': ['replay-A.json', 'replay-B.json',
                                              'eligibility.json', 'cuda.json']})
        for name in members[-1]['required_process_receipt_names']:
            publish(root / name, {'returncode': 0, 'finished_epoch': 1, 'deadline_epoch': 2})
    return {'schema': 'prospective-own45-completion-cohort-v1',
            'status': 'frozen-before-both-formal-executions', 'slots': [4, 5],
            'latency_order': [4, 5], 'completion_deadline_epoch': 1791162500,
            'members': members}


def test_solo_is_unmodified_and_source_coordinator_binding_fail_closed(tmp_path):
    assert cohort.bind({}, 4, '4' * 64, sha) is None
    c = fixture(tmp_path)
    path = tmp_path / 'cohort.json'
    publish(path, c)
    config = {'root': str(tmp_path / '4'),
              'cohort': {'manifest': str(path), 'manifest_sha256': sha(path)}}
    assert cohort.bind(config, 4, '4' * 64, sha) == c
    with pytest.raises(AssertionError):
        cohort.bind(config, 4, '5' * 64, sha)


def test_follower_cannot_latency_before_leader_and_both_final_release_requires_both(tmp_path):
    c = fixture(tmp_path)
    for slot in (4, 5):
        owner = cohort.member(c, slot)
        cohort.publish_ready(c, slot, owner['root'],
                             owner['required_process_receipt_names'], sha, publish)
    cohort.before_latency(c, 4, wait, sha)
    with pytest.raises(FileNotFoundError):
        cohort.before_latency(c, 5, wait, sha)
    publish(tmp_path / '4' / 'latency-process-result.json',
            {'returncode': 0, 'finished_epoch': 1, 'deadline_epoch': 2})
    with pytest.raises(FileNotFoundError):
        cohort.after_latency(c, 4, tmp_path / '4', wait, sha, publish)
    cohort.before_latency(c, 5, wait, sha)
    publish(tmp_path / '5' / 'latency-process-result.json',
            {'returncode': 0, 'finished_epoch': 1, 'deadline_epoch': 2})
    cohort.after_latency(c, 5, tmp_path / '5', wait, sha, publish)
    for slot in (4, 5):
        cohort.verify_receipt(c, slot, 'latency', wait, sha)


def test_tampered_or_failed_peer_process_rejects_without_reading_outcomes(tmp_path):
    c = fixture(tmp_path)
    owner = cohort.member(c, 5)
    cohort.publish_ready(c, 5, owner['root'], owner['required_process_receipt_names'], sha, publish)
    path = tmp_path / '5' / 'cuda.json'
    path.write_text(json.dumps({'returncode': 1, 'finished_epoch': 1, 'deadline_epoch': 2}))
    with pytest.raises(AssertionError):
        cohort.verify_receipt(c, 5, 'ready', wait, sha)
