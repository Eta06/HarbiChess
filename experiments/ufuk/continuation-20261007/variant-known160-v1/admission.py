"""Typed own-Q, TD0.5, or closed-terminal MC admission; unknown phases fail closed."""

import contextlib
import importlib.util
import json
import random
import sys
import types

from collection_admission import collection_audits
from original_ownq_admission import SEARCH_SHA as SEARCH_SHA
from original_ownq_admission import SEEDS
from original_ownq_admission import child as ownq_child
from original_ownq_admission import import_nnue as import_nnue
from teacher_admission import module, pin, read, sha, teacher

TD_PHASE = 'own-learning-tdlambda-v1'
TD_STATUS = 'PASS-tdlambda-fixed-phase-and-fresh-native-loads-not-strength'
TD_NATIVE = 'own-kingbucket-nnue16-tdlambda-full-native-cpu-v1'
VARIANTS = {
    'tdlambda-0.5-v1': dict(phase=TD_PHASE, native=TD_NATIVE, status=TD_STATUS,
        registration='NNUE-own-tdlambda-training-orchestration-v1',
        proof_key='tdlambda_proof_result',
        modules=('model', 'native', 'contract', 'tdlambda_targets', 'prepare', 'contract_builder')),
    'closed-terminal-mc-v1': dict(phase='own-closed-terminal-learning-v1',
        native='own-kingbucket-nnue16-closed-terminal-full-native-cpu-v1',
        status='PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength',
        registration='NNUE-own-closed-terminal-training-orchestration-v1',
        proof_key='closed_terminal_proof_result',
        modules=('model', 'native', 'contract', 'convert', 'contract_builder')),
}


def variant_definition(phase):
    matches = [v for v in VARIANTS.values() if v['phase'] == phase]
    if len(matches) != 1:
        raise ValueError('unknown variant phase; no legacy native relabel')
    return matches[0]



@contextlib.contextmanager
def td_modules(info, contract):
    refs = info['variant_helpers']
    names = variant_definition(contract['phase'])['modules']
    if set(refs) != set(names):
        raise ValueError('exact TD native/model/admission/target/replay helper closure')
    for ref in refs.values():
        path = str(pin(ref).resolve())
        maps = [contract.get(k, {}) for k in ('source_sha256', 'execution_helpers_sha256')]
        if not any(m.get(path) == ref['sha256'] for m in maps):
            raise ValueError('TD helper must be exact original contract source')
    previous = {n: sys.modules.get(n) for n in names}
    loaded = {}
    try:
        for name in names:
            path = pin(refs[name])
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def compare_contracts(proof, fit, info):
    proof_key = variant_definition(fit['phase'])['proof_key']
    phase_keys = {'execution_mode', 'original_first_epoch', 'original_deadline_epoch',
                  'contract_build_seal_sha256', proof_key}
    if {k: v for k, v in proof.items() if k not in phase_keys} != {
            k: v for k, v in fit.items() if k not in phase_keys}:
        raise ValueError('ALL TD proof/fit dataset/raw/source/math/parent fields must match')
    if (proof['execution_mode'] != 'proof' or fit['execution_mode'] != 'fresh-fit'
            or proof_key in proof
            or fit[proof_key] != info['proof_result']):
        raise ValueError('proof then fresh64 named-proof transition only')


def td_receipt(result, registration, ref, contract, steps, end):
    definition = variant_definition(contract['phase'])
    mode = 'proof' if len(steps) == 6 else 'fresh-fit'
    cap = 600 if mode == 'proof' else 1800
    if (result['status'] != definition['status'] or result['mode'] != mode
            or result['seed'] != contract['seed'] or result['own_updates'] != steps[-1]
            or result['contract_sha256'] != ref['sha256']
            or result['dataset_sha256'] != contract['dataset_sha256']
            or result['target_provenance_sha256'] != contract['target_provenance_sha256']
            or result['teacher_labels_used'] is not False
            or result['raw_zip_identity_claimed'] is not False
            or result['weights_only_initializer']['sha256'] != contract['parent_candidate_sha256']
            or not result['first'] < result['finished'] <= result['deadline'] <= end
            or result['deadline'] > result['first'] + cap
            or (result['first'], result['deadline']) !=
            (contract['original_first_epoch'], contract['original_deadline_epoch'])
            or registration['schema'] != definition['registration']
            or registration['mode'] != mode or registration['seed'] != contract['seed']
            or (registration['first'], registration['deadline']) !=
            (result['first'], result['deadline'])
            or registration['contract'] != ref
            or registration['dataset']['sha256'] != contract['dataset_sha256']
            or registration['target_provenance']['sha256'] != contract['target_provenance_sha256']
            or registration['parent_candidate']['sha256'] != contract['parent_candidate_sha256']
            or (mode == 'proof' and result['full_payload_bits_equal'] is not True)):
        raise ValueError('actual TD-specific completed original-clock receipt/registration')
    commands = result['commands']
    if len(commands) != (9 if mode == 'proof' else 3) or any(
            c['returncode'] != 0 or c['finished'] > result['deadline'] for c in commands):
        raise ValueError('ALL whole/pause/freshresume/fresh-load actual child commands')
    audits = [c for c in commands if '--audit-only' in c['command']]
    payloads = result['native_payloads']
    if len(audits) != len(steps) or [r['step'] for r in payloads] != steps:
        raise ValueError('exact six proof plus two fresh-fit strict load inventory')
    for command, payload, step in zip(audits, payloads, steps, strict=True):
        argv = command['command']
        path = pin(payload)
        if (payload['actual_fresh_process'] is not True
                or argv[1] != registration['train']['path']
                or argv[argv.index('--dataset') + 1] != registration['dataset']['path']
                or argv[argv.index('--resume') + 1] != str(path)
                or argv[argv.index('--resume-sha256') + 1] != payload['sha256']
                or argv[argv.index('--contract') + 1] != ref['path']
                or int(argv[argv.index('--stop') + 1]) != step):
            raise ValueError('actual fresh-load exact native/contract/counter command')
        log = pin(dict(path=payload['log_path'], sha256=command['log_sha256']))
        if json.loads(log.read_bytes()) != dict(status='PASS-strict-native-readonly', step=step):
            raise ValueError('actual SHA-sealed strict fresh-load stdout')
    return payloads


def replay_targets(info, c, modules):
    if c['phase'] == 'own-closed-terminal-learning-v1':
        provenance = read(info['target_provenance'])
        if provenance['inputs'] != c['raw_collection_inputs']:
            raise ValueError('actual MC raw input/spec source lineage')
        data, trace = modules['convert'].convert(provenance['inputs'])
        if (data != pin(info['dataset']).read_bytes()
                or trace != pin(info['target_provenance']).read_bytes()):
            raise ValueError('ALL MC history/terminal/UNKNOWN/protected/feature bytes differ')
        return
    inputs = info['td_conversion_inputs']
    expected = {'source_dataset', 'source_provenance', 'collection_receipt', 'events',
                'conversion_result', 'build_seal'}
    if set(inputs) != expected:
        raise ValueError('exact TD target transformation input closure')
    raw = {k: pin(v).read_bytes() for k, v in inputs.items()}
    seal = json.loads(raw['build_seal'])
    provenance = json.loads(pin(info['target_provenance']).read_bytes())
    if (provenance['build_seal_sha256'] != sha(pin(inputs['build_seal']))
            or seal['seed'] != c['seed']
            or seal['raw_collection_inputs'] != c['raw_collection_inputs']
            or seal['inputs'] != {k: sha(pin(v)) for k, v in inputs.items() if k != 'build_seal'}):
        raise ValueError('TD original raw collection and build seal source lineage')
    source_provenance = json.loads(raw['source_provenance'])
    source_converter = module(info['source_converter'], 'variant_actual_original_own_converter')
    source_data, source_trace = source_converter.convert(source_provenance['inputs'])
    if source_data != raw['source_dataset'] or source_trace != raw['source_provenance']:
        raise ValueError('original own-Q converter/fullhistory/feature/prior/alias replay mismatch')
    conversion = json.loads(raw['conversion_result'])
    if (not conversion['first'] <= conversion['finished'] <= conversion['deadline']
            <= conversion['first'] + 600):
        raise ValueError('actual original own-Q conversion inclusive600 receipt')
    converter = modules['prepare']
    # Historical pure replay ONLY: preserve old clocks; no experiment clock extension.
    # Replay target function's recorded-time admission at its original midpoint.
    previous_time = converter.time
    converter.time = types.SimpleNamespace(time=lambda: (seal['first'] + seal['deadline']) / 2)
    try:
        data, trace = converter.build_tdlambda_dataset(**{k + '_bytes': v for k, v in raw.items()})
    finally:
        converter.time = previous_time
    if (data != pin(info['dataset']).read_bytes()
            or trace != pin(info['target_provenance']).read_bytes()):
        raise ValueError('actual fullhistory/mover/UNKNOWN/TD0.5 target bytes differ')


def child(protocol, seed, native):
    collection_audits(protocol, read(protocol['collection_audit_set']))
    variant = protocol['target_variant']
    if variant == 'own-q-v2':
        return ownq_child(protocol, seed, native)
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError('unsupported or unqualified MC/other variant; no phase relabel')
    import torch

    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    try:
        parent_info, info = protocol['teachers'][str(seed)], protocol['children'][str(seed)]
        parent, _, parent_result = teacher(parent_info, native)
        definition = VARIANTS[variant]
        c, proof = read(info['contract']), read(info['proof_contract'])
        if (c['phase'] != definition['phase'] or c['native_schema'] != definition['native']
                or (variant == 'tdlambda-0.5-v1' and c.get('lambda') != .5)
                or (variant == 'closed-terminal-mc-v1' and 'lambda' in c)
                or c['seed'] != seed or c['updates'] != 64
                or c['bootstrap_candidate_sha256'] != parent_info['candidate']['sha256']
                or c['bootstrap_candidate_path'] != parent_info['candidate']['path']
                or protocol['models'][str(seed)]['parent'] != parent_info['candidate']
                or protocol['models'][str(seed)]['learned'] != info['candidate']
                or info['target_provenance']['sha256'] != c['target_provenance_sha256']
                or info['dataset']['sha256'] != c['dataset_sha256']):
            raise ValueError('separate typed TD64 same exact teacher-parent endpoint')
        compare_contracts(proof, c, info)
        for contract in (proof, c):
            for field in ('source_sha256', 'execution_helpers_sha256', 'inference_source_sha256'):
                if not contract[field]:
                    raise ValueError('nonempty immutable TD source/inference closure')
                for path, digest in contract[field].items():
                    pin(dict(path=path, sha256=digest))
        with td_modules(info, c) as modules:
            td_native = modules['native']
            for mode, original in (('proof', proof), ('fit', c)):
                seal_ref = info['contract_build_seals'][mode]
                seal = read(seal_ref)
                if seal_ref['sha256'] != original['contract_build_seal_sha256']:
                    raise ValueError('actual original ROOT contract build seal bytes')
                seal['seal_sha256'] = seal_ref['sha256']
                if modules['contract_builder'].make_contract(seal) != original:
                    raise ValueError('full contract metadata must rederive from original seal')
            if c['math'] != td_native.MATH or definition['native'] != td_native.SCHEMA:
                raise ValueError('actual TD native math/schema')
            modules['contract'].admit_contract(pin(info['contract']).read_bytes(),
                                               pin(info['dataset']).read_bytes(),
                                               pin(info['target_provenance']).read_bytes())
            replay_targets(info, c, modules)
            states_by_mode = {}
            checks = []
            for contract, mode, steps in ((proof, 'proof', [0, 8, 0, 4, 4, 8]),
                                           (c, 'fit', [0, 64])):
                result_ref = info[mode + '_result']
                result = read(result_ref)
                registration = read(info[mode + '_registration'])
                if result['registration_sha256'] != info[mode + '_registration']['sha256']:
                    raise ValueError('actual ROOT phase registration SHA')
                rows = td_receipt(result, registration, info['proof_contract' if mode == 'proof'
                                  else 'contract'], contract, steps,
                                  protocol['ROOToperator_end_epoch'])
                states = [td_native.load_native(pin(r), contract).native() for r in rows]
                if any(s['step'] != n for s, n in zip(states, steps, strict=True)):
                    raise ValueError('actual strict native counter')
                if mode == 'proof' and not all(td_native.bits_equal(states[a], states[b])
                                               for a, b in ((0, 2), (1, 5), (3, 4))):
                    raise ValueError('whole8/pause4/freshresume8 ALL model/Adam/RNG bits')
                states_by_mode[mode] = states
                checks.extend(rows)
            zero, final = states_by_mode['fit']
            fit_rows = read(info['fit_result'])['native_payloads']
            if any(r['path'] != info[k]['path'] or r['sha256'] != info[k]['sha256']
                   for r, k in zip(fit_rows, ('initial', 'native'), strict=True)):
                raise ValueError('exact fresh0/fixed-final64 admitted files')
            candidate = torch.load(pin(info['candidate']), map_location='cpu', weights_only=False)
            if (zero['optimizer']['state'] or not td_native.bits_equal(zero['model'], parent)
                    or not td_native.bits_equal(zero['baseline'], parent)
                    or not td_native.bits_equal(final['baseline'], parent)
                    or td_native.bits_equal(final['model'], parent)
                    or candidate.keys() != {'schema', 'contract', 'model'}
                    or candidate['schema'] != 'own-kingbucket-nnue16-model-v1'
                    or candidate['contract'] != c
                    or not td_native.bits_equal(candidate['model'], final['model'])):
                raise ValueError('fresh teacher WEIGHTS, Adam/baseline/changed64/finalcandidate')
            return (candidate['model'], parent, dict(seed=seed, teacher=parent_result,
                    own_updates=64, own_proof_fresh_loads=6, own_fit_fresh_loads=2,
                    full_native_checks=len(checks),
                    own_candidate_sha256=info['candidate']['sha256'],
                    teacher_ancestry=True, target_variant=variant, historical_target_replay=True))
    finally:
        random.setstate(before_python)
        torch.set_rng_state(before_torch)
