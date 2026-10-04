"""Prospectively pinned outcome-blind completion barriers; no jobs or deadline resets."""
import json
import time
from pathlib import Path

SOURCES = {4: 'a278bba67bce962cb9294f0d24040e02e9acf1f4',
           5: '4515a7c0dda3b4f9615c2fc78a47c872ab14699d'}


def bind(config, slot, coordinator_sha, sha):
    if 'cohort' not in config:
        return None
    item = config['cohort']
    assert sha(item['manifest']) == item['manifest_sha256']
    manifest = json.loads(Path(item['manifest']).read_text())
    assert manifest['schema'] == 'prospective-own45-completion-cohort-v1'
    assert manifest['status'] == 'frozen-before-both-formal-executions'
    assert manifest['slots'] == [4, 5] and manifest['latency_order'] == [4, 5]
    assert manifest['completion_deadline_epoch'] > 0
    assert manifest['completion_deadline_epoch'] + 7300 + 120 + 120 + 180 + 120 < 1791170400
    for member in manifest['members']:
        assert member['slot'] in SOURCES
        assert member['source_commit'] == SOURCES[member['slot']]
        assert len(member['coordinator_sha256']) == 64
        assert Path(member['root']).is_absolute()
    assert {m['slot'] for m in manifest['members']} == {4, 5}
    assert len(manifest['members']) == 2
    own = next(m for m in manifest['members'] if m['slot'] == slot)
    assert own['coordinator_sha256'] == coordinator_sha
    assert Path(own['root']).resolve() == Path(config['root']).resolve()
    return manifest


def member(cohort, slot):
    return next(m for m in cohort['members'] if m['slot'] == slot)


def publish_ready(cohort, slot, root, receipts, sha, publish, artifacts=None):
    owner = member(cohort, slot)
    expected = set(owner['required_process_receipt_names'])
    assert set(receipts) == expected
    proof = {}
    for name in receipts:
        assert Path(name).name == name
        path = Path(root) / name
        data = json.loads(path.read_text())
        assert data['returncode'] == 0
        assert data['finished_epoch'] <= data['deadline_epoch']
        proof[name] = sha(path)
    publish(Path(root) / 'cohort-ready.json', {
        'schema': 'own45-prelatency-completion-receipt-v1',
        'slot': slot, 'source_commit': SOURCES[slot],
        'coordinator_sha256': owner['coordinator_sha256'],
        'process_receipt_sha256': proof,
        'validated_artifact_sha256': {str(path): sha(path) for path in (artifacts or {})},
        'finished_epoch': time.time()})


def verify_receipt(cohort, slot, kind, wait_json, sha):
    owner = member(cohort, slot)
    root = Path(owner['root'])
    name = 'cohort-ready.json' if kind == 'ready' else 'cohort-latency-complete.json'
    data = wait_json(root / name, cohort['completion_deadline_epoch'])
    assert data['slot'] == slot and data['source_commit'] == SOURCES[slot]
    assert data['coordinator_sha256'] == owner['coordinator_sha256']
    assert data['finished_epoch'] <= cohort['completion_deadline_epoch']
    assert data['schema'] == ('own45-prelatency-completion-receipt-v1' if kind == 'ready'
                              else 'own45-latency-owner-completion-receipt-v1')
    if kind == 'ready':
        for artifact, digest in data['validated_artifact_sha256'].items():
            assert sha(artifact) == digest
    mapping = data['process_receipt_sha256']
    expected = (set(owner['required_process_receipt_names']) if kind == 'ready'
                else {'latency-process-result.json'})
    assert set(mapping) == expected
    for filename, digest in mapping.items():
        assert Path(filename).name == filename
        assert sha(root / filename) == digest
        result = wait_json(root / filename, cohort['completion_deadline_epoch'])
        assert result['returncode'] == 0
        assert result['finished_epoch'] <= result['deadline_epoch']
    return data


def before_latency(cohort, slot, wait_json, sha):
    if cohort is None:
        return
    for candidate in (4, 5):
        verify_receipt(cohort, candidate, 'ready', wait_json, sha)
    if slot == 5:
        verify_receipt(cohort, 4, 'latency', wait_json, sha)


def after_latency(cohort, slot, root, wait_json, sha, publish):
    if cohort is None:
        return
    owner = member(cohort, slot)
    path = Path(root) / 'latency-process-result.json'
    result = wait_json(path, cohort['completion_deadline_epoch'])
    assert result['returncode'] == 0 and result['finished_epoch'] <= result['deadline_epoch']
    publish(Path(root) / 'cohort-latency-complete.json', {
        'schema': 'own45-latency-owner-completion-receipt-v1',
        'slot': slot, 'source_commit': SOURCES[slot],
        'coordinator_sha256': owner['coordinator_sha256'],
        'process_receipt_sha256': {path.name: sha(path)}, 'finished_epoch': time.time()})
    for candidate in (4, 5):
        verify_receipt(cohort, candidate, 'latency', wait_json, sha)
