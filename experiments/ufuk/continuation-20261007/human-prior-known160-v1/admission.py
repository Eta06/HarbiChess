"""Human-prior generation admission; never teacher256 or anonymous own64."""
import contextlib
import hashlib
import importlib.util
import json
import random
import sys

from original_ownq_admission import SEARCH_SHA as SEARCH_SHA
from original_ownq_admission import SEEDS as SEEDS
from original_ownq_admission import import_nnue as import_nnue
from teacher_admission import pin, read

VARIANT = 'human-prior-own-q-v1'
NAMES = ('model', 'parent_bridge', 'zero_parent', 'native', 'convert', 'specs', 'contracts')
ZERO = 'human-prior-zero-residual-init-v1'


def check_identity(q, seed, c, parent, parent_ref):
    if (q['target_variant'] != VARIANT or seed not in SEEDS
            or c['execution_scope_schema'] != 'human-prior-own-execution-contract-v1'
            or c['phase'] != 'own-learning' or c['updates'] != 64
            or c['seed'] != seed or parent['seed'] != seed
            or c['generation'] != parent['generation'] + 1
            or c['parent_generation'] != parent['generation']
            or c['lineage_origin'] != parent['lineage_origin']
            or c['search_helper'] != parent['search_helper']
            or c['search_helper'] != q['original_search']
            or c['parent_candidate'] != parent_ref
            or c['bootstrap_candidate_sha256'] != parent_ref['sha256']
            or c['bootstrap_candidate_path'] != parent_ref['path']
            or q['models'][str(seed)]['parent'] != parent_ref
            or q['models'][str(seed)]['learned'] != q['children'][str(seed)]['candidate']
            or c['teacher_labels_used_in_own_phase'] is not False):
        raise ValueError('exact current admitted human/own generation parent and fixed child64')
    if ((parent['generation'] == 0 and
         (parent['phase'] != ZERO or parent['updates'] != 0))
            or (parent['generation'] > 0 and
                (parent['phase'] != 'own-learning' or parent['updates'] != 64))):
        raise ValueError('literal zero0 or exact previous own64, never teacher256')


@contextlib.contextmanager
def human_modules(info, c):
    refs = info['human_helpers']
    if set(refs) != set(NAMES):
        raise ValueError('exact human native/parent/converter/audit/build dependency closure')
    old = {n: sys.modules.get(n) for n in NAMES}
    loaded = {}
    try:
        for name in NAMES:
            ref = refs[name]
            path = pin(ref)
            if not any(m.get(str(path.resolve())) == ref['sha256'] for m in
                       (c['source_sha256'], c['execution_helpers_sha256'])):
                raise ValueError('actual original human helper source SHA')
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            loaded[name] = mod
        yield loaded
    finally:
        for name, mod in old.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def receipt(result, reg, contract_ref, c, steps, end, bridge, parent_contract_sha256):
    mode = 'proof' if len(steps) == 6 else 'fresh-fit'
    parent_binding = dict(c['parent_candidate'], contract_sha256=parent_contract_sha256)
    if (result['status'] != 'PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength'
            or result['mode'] != mode or result['own_updates'] != steps[-1]
            or result['generation'] != c['generation']
            or result['contract_sha256'] != contract_ref['sha256']
            or result['raw_zip_identity_claimed'] is not False
            or result['teacher_labels_used'] is not False
            or result['weights_only_initializer'] != parent_binding
            or not result['first'] < result['finished'] <= result['deadline'] <= end
            or result['deadline'] > result['first'] + (600 if mode == 'proof' else 1800)
            or (result['first'], result['deadline']) !=
                (c['original_first_epoch'], c['original_deadline_epoch'])
            or reg['schema'] != 'human-prior-own-training-orchestration-v1'
            or reg['mode'] != mode or reg['seed'] != c['seed']
            or (reg['first'], reg['deadline']) != (result['first'], result['deadline'])
            or reg['contract'] != contract_ref
            or reg['dataset']['sha256'] != c['dataset_sha256']
            or reg['parent_candidate'] != parent_binding
            or (mode == 'proof' and result['full_payload_bits_equal'] is not True)):
        raise ValueError('actual same-data new human own phase receipt, not synthetic/old clocks')
    commands = result['commands']
    if len(commands) != (9 if mode == 'proof' else 3) or any(
            x['returncode'] != 0 or not result['first'] <= x['finished'] <= result['deadline']
            for x in commands):
        raise ValueError('all actual optimizer/native-load commands')
    bridge.native_rows(result, steps)
    audits = [x for x in commands if '--audit-only' in x['command']]
    for owner, _row in zip(audits, result['native_payloads'], strict=True):
        argv = owner['command']
        if (argv[1] != reg['train']['path']
                or argv[argv.index('--contract') + 1] != contract_ref['path']
                or argv[argv.index('--dataset') + 1] != reg['dataset']['path']):
            raise ValueError('actual original helper/contract/data load argv')
    return result['native_payloads']


def child(q, seed, _original_architecture_native):
    import torch

    before_py, before_torch = random.getstate(), torch.get_rng_state().clone()
    try:
        info = q['children'][str(seed)]
        c, proof = read(info['contract']), read(info['proof_contract'])
        for contract in (c, proof):
            for field in ('source_sha256', 'execution_helpers_sha256', 'inference_source_sha256'):
                if not contract[field]:
                    raise ValueError('nonempty all source/data/inference closure')
                for path, digest in contract[field].items():
                    pin(dict(path=path, sha256=digest))
        with human_modules(info, c) as modules:
            bridge, native = modules['parent_bridge'], modules['native']
            parent_row = q['parents'][str(seed)]
            spec, parent_c = bridge.validate_admission_result(
                parent_row['admission_seal'], parent_row['admission_result'])
            parent = bridge.admitted_candidate(parent_row['admission_seal'],
                                               parent_row['admission_result'], torch)['model']
            # Reopen exact parent ORIGINAL native (possibly earlier generation source),
            # rather than pretending current child's native schema proves parent state.
            parent_native_ref = spec['parent_native_helper']
            parent_model_ref = spec['parent_model_helper']
            previous_model = sys.modules['model']
            try:
                for ref, name in ((parent_model_ref, 'model'),
                                  (parent_native_ref, 'human_original_parent_native')):
                    path = pin(ref)
                    desc = importlib.util.spec_from_file_location(name, path)
                    mod = importlib.util.module_from_spec(desc)
                    sys.modules[name] = mod
                    desc.loader.exec_module(mod)
                parent, parent_result = bridge.admit(spec, mod, lambda path:
                    torch.load(path, map_location='cpu', weights_only=False))
                if parent_c['generation'] == 0:
                    root_result = read(spec['parent_initialization_result'])
                    parent_states = [mod.load_native(pin(r), parent_c).native()
                                     for r in root_result['native_payloads']]
                    if len(parent_states) != 2 or not mod.bits_equal(*parent_states):
                        raise ValueError('both actual literal-zero original-native opens')
                    parent_checks = 2
                else:
                    parent_proof = read(spec['parent_proof_contract'])
                    parent_states = {}
                    for mode, contract in (('proof', parent_proof), ('fit', parent_c)):
                        result = read(spec['parent_' + mode + '_result'])
                        parent_states[mode] = [mod.load_native(pin(r), contract).native()
                                              for r in result['native_payloads']]
                    if not all(mod.bits_equal(parent_states['proof'][a],
                                              parent_states['proof'][b])
                               for a, b in ((0, 2), (1, 5), (3, 4))):
                        raise ValueError('previous-own parent actual all-payload proof chain')
                    if (parent_states['fit'][0]['optimizer']['state']
                            or not mod.bits_equal(parent_states['fit'][1]['model'], parent)):
                        raise ValueError('previous own parent fresh0/final64 storage')
                    parent_checks = 8
            finally:
                sys.modules['model'] = previous_model
                sys.modules.pop('human_original_parent_native', None)
            check_identity(q, seed, c, parent_c, spec['parent_candidate'])
            if (c['parent_admission_seal'] != parent_row['admission_seal']
                    or c['parent_admission_result'] != parent_row['admission_result']
                    or c['math'] != native.MATH
                    or native.SCHEMA != 'human-prior-own-nnue16-native-cpu-v1'):
                raise ValueError('named parent admission and new typed human native')
            for mode, contract in (('proof', proof), ('fit', c)):
                seal = read(info['contract_build_seals'][mode])
                if seal != contract['contract_build_seal']:
                    raise ValueError('exact original ROOT build seal')
                if modules['contracts'].build(seal) != contract:
                    raise ValueError('all contract fields rederived from actual inputs')
                if (contract['parent_candidate'] != c['parent_candidate']
                        or contract['dataset_sha256'] != c['dataset_sha256']
                        or contract['target_provenance_sha256'] != c['target_provenance_sha256']
                        or contract['generation'] != c['generation']):
                    raise ValueError('proof/fit exact parent, data, target and generation')
            if c['own_phase_proof'] != info['proof_result']:
                raise ValueError('fresh64 depends on exact completed actual own8 proof')
            trace = read(info['target_provenance'])
            data, provenance = modules['convert'].convert(trace['inputs'])
            if (data != pin(info['dataset']).read_bytes()
                    or provenance != pin(info['target_provenance']).read_bytes()
                    or trace['inputs'] != c['raw_collection_inputs']):
                raise ValueError('all1024 raw histories/aliases/ownQ target bytes')
            seal = c['contract_build_seal']
            if seal['collection_replay_audit_set'] != q['collection_audit_set']:
                raise ValueError('actual both-seed twelve-search qualification')
            modules['specs'].require_audits(pin(q['collection_audit_set']),
                json.loads(pin(seal['collection_registrations']).read_bytes()))
            states = {}
            for mode, contract, steps in (('proof', proof, [0, 8, 0, 4, 4, 8]),
                                          ('fit', c, [0, 64])):
                result, reg = read(info[mode + '_result']), read(info[mode + '_registration'])
                if result['registration_sha256'] != info[mode + '_registration']['sha256']:
                    raise ValueError('exact registered owned phase')
                contract_ref = info['proof_contract' if mode == 'proof' else 'contract']
                rows = receipt(result, reg, contract_ref,
                               contract, steps, q['ROOToperator_end_epoch'], bridge,
                               hashlib.sha256(bridge.canonical(parent_c)).hexdigest())
                states[mode] = [native.load_native(pin(row), contract).native() for row in rows]
                if [s['step'] for s in states[mode]] != steps:
                    raise ValueError('strict counters')
                if mode == 'proof' and not all(native.bits_equal(states[mode][a], states[mode][b])
                                              for a, b in ((0, 2), (1, 5), (3, 4))):
                    raise ValueError('whole8/pause4/fresh8 all-model/Adam/global/sampler RNG')
            zero, final = states['fit']
            for row, key in zip(read(info['fit_result'])['native_payloads'],
                                ('initial', 'native'), strict=True):
                if {k: row[k] for k in ('path', 'sha256')} != info[key]:
                    raise ValueError('exact production initial0/final64 endpoints')
            candidate = torch.load(pin(info['candidate']), map_location='cpu', weights_only=False)
            if (zero['optimizer']['state'] or not native.bits_equal(zero['model'], parent)
                    or not native.bits_equal(zero['baseline'], parent)
                    or not native.bits_equal(final['baseline'], parent)
                    or native.bits_equal(final['model'], parent)
                    or set(candidate) != {'schema', 'contract', 'model'}
                    or candidate['schema'] != 'own-kingbucket-nnue16-model-v1'
                    or candidate['contract'] != c
                    or not native.bits_equal(candidate['model'], final['model'])):
                raise ValueError('fresh current-parent weights-only/changed64/native candidate')
            return candidate['model'], parent, dict(seed=seed, parent=parent_result,
                teacher_ancestry=False, parent_full_native_checks=parent_checks,
                parent_generation=parent_c['generation'],
                own_generation=c['generation'], own_updates=64, own_proof_fresh_loads=6,
                own_fit_fresh_loads=2, target_variant=VARIANT)
    finally:
        random.setstate(before_py)
        torch.set_rng_state(before_torch)
