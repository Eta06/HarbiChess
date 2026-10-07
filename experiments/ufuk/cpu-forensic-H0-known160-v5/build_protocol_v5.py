"""Build a v5 H0-parent protocol from frozen forensic-v4 proof/fit outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (20262905, 20262906)
BASE = Path('/workspace/work/harbichess/continuation-20261007')
NATIVE_ROOT = Path('/dev/shm/harbichess-human-randomstarts-forensic-native-v4')
DATA_ROOT = Path('/dev/shm/harbichess-human-randomstarts-forensic-data-v4')
AUDIT_SET = BASE / 'procedural-forensic-v4-actual-audit-set.json'
AUDIT_VALIDATOR = BASE / 'human-randomstarts-own-v3-forensic-shadow/forensic_audit_set.py'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    path = path.resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def build(base_protocol: Path) -> dict:
    q = read(base_protocol)
    q['target_variant'] = 'forensic-own-v5-h0'
    q.pop('teachers', None)
    q['collection_audit_set'] = ref(AUDIT_SET)
    q['collection_audit_validator'] = ref(
        BASE / 'human-randomstarts-own-v3-forensic-shadow/forensic_audit_set.py')
    q['parents'] = {}
    q['children'] = {}
    for seed in SEEDS:
        key = str(seed)
        seed_dir = BASE / f'forensic-own-v4-actual-{seed}'
        proof_outer = read(seed_dir / 'proof/result.json')
        fit_outer = read(seed_dir / 'fresh-fit/result.json')
        proof_contract_ref = ref(Path(proof_outer['contract']['path']))
        fit_contract_ref = ref(Path(fit_outer['contract']['path']))
        proof_contract = read(Path(proof_contract_ref['path']))
        fit_contract = read(Path(fit_contract_ref['path']))
        if (proof_outer['status'] != 'PASS-forensic-v4-whole-pause-fresh-native-proof-not-strength'
                or fit_outer['status'] != 'PASS-forensic-v4-fresh64-native-loads-not-strength'
                or proof_outer['mode'] != 'proof' or fit_outer['mode'] != 'fresh-fit'
                or proof_outer['seed'] != seed or fit_outer['seed'] != seed
                or proof_contract['phase'] != 'own-learning'
                or fit_contract['phase'] != 'own-learning'
                or proof_contract['parent_generation'] != 0
                or fit_contract['parent_generation'] != 0
                or proof_contract['parent_candidate'] != fit_contract['parent_candidate']):
            raise ValueError('actual two-phase forensic native proof/fit and literal H0 binding')
        parent = {
            'candidate': fit_contract['parent_candidate'],
            'native': fit_contract['parent_native'],
            'contract': fit_contract['parent_ancestry']['parent_contract'],
            'admission_seal': fit_contract['parent_admission_seal'],
            'admission_result': fit_contract['parent_admission_result'],
        }
        proof_inner = ref(Path(proof_outer['phase_result']['path']))
        fit_inner = ref(Path(fit_outer['phase_result']['path']))
        proof_registration = ref(seed_dir / 'proof/training-registration.json')
        fit_registration = ref(seed_dir / 'fresh-fit/training-registration.json')
        proof_result = read(Path(proof_inner['path']))
        fit_result = read(Path(fit_inner['path']))
        if (proof_result['registration_sha256'] != proof_registration['sha256']
                or fit_result['registration_sha256'] != fit_registration['sha256']):
            raise ValueError('inner phase receipts bind the training registrations')
        raw = fit_contract['raw_collection_inputs']
        child = {
            'candidate': ref(Path(fit_outer['candidate']['path'])),
            'contract': fit_contract_ref,
            'proof_contract': proof_contract_ref,
            'dataset': ref(Path(fit_outer['dataset']['path'])),
            'target_provenance': ref(Path(fit_outer['provenance']['path'])),
            'proof_result': proof_inner,
            'fit_result': fit_inner,
            'outer_proof_result': ref(seed_dir / 'proof/result.json'),
            'outer_fit_result': ref(seed_dir / 'fresh-fit/result.json'),
            'proof_registration': proof_registration,
            'fit_registration': fit_registration,
            'contract_build_seals': {
                'proof': ref(seed_dir / 'proof/contract-build-seal.json'),
                'fit': ref(seed_dir / 'fresh-fit/contract-build-seal.json'),
            },
            'initial': fit_result['native_payloads'][0],
            'native': fit_result['native_payloads'][1],
            'collection_registration': raw['registration'],
            'collection_receipt': raw['receipt'],
            'events': raw['events'],
            'forensic_view': raw['forensic_view'],
            'variant_helpers': {
                'model': ref(BASE / 'human-randomstarts-own-v4-forensic-native/model.py'),
                'native': ref(BASE / 'human-randomstarts-own-v4-forensic-native/native.py'),
                'contract': ref(BASE / 'human-randomstarts-own-v4-forensic-native'
                                 / 'contracts_forensic_v4.py'),
                'convert': ref(BASE / 'human-randomstarts-own-v4-forensic-native'
                               / 'convert_forensic_v3.py'),
                'contract_builder': ref(BASE / 'human-randomstarts-own-v4-forensic-native'
                                        / 'contracts_forensic_v4.py'),
            },
        }
        q['parents'][key] = parent
        q['children'][key] = child
        q['models'][key]['parent'] = parent['candidate']
        q['models'][key]['learned'] = child['candidate']
    return q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--base-protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = (json.dumps(build(a.base_protocol), sort_keys=True,
                       separators=(',', ':')).encode() + b'\n')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(data)
    print(json.dumps({'path': str(a.output), 'sha256': hashlib.sha256(data).hexdigest(),
                      'bytes': len(data)}, sort_keys=True))


if __name__ == '__main__':
    main()
