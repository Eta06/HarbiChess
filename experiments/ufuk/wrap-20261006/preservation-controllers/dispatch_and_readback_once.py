"""ROOT-owned one parts workflow, then full anonymous independent byte readback."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STAGE = Path('/workspace/HarbiChess/experiments/ufuk/nnue-own-state-blob-v9')
COMMON = HERE.parent / 'fresh-state-v8-dispatch-v1/dispatch_once_v8.py'
spec = importlib.util.spec_from_file_location('owned_common', COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
reg = json.loads((HERE / 'dispatch-registration.json').read_bytes())
assert common.digest(Path(__file__)) == reg['controller_sha256']
assert common.digest(COMMON) == reg['common_controller_sha256']
common.FIRST, common.END = reg['first'], reg['deadline']
common.HEAD = reg['head']
common.TITLE = 'NNUE V9 parts / ' + reg['parts_control_sha256'][:12]
common.WORKFLOW = 'nnue-own-state-parts-v9.yml'
sys.path.insert(0, str(STAGE))
import raw_capsule_v9 as codec
import release_transport_v9 as release
import parts_receiver_v1 as parts

c = json.loads((STAGE / 'parts-control.json').read_bytes())
mraw = (STAGE / 'manifest.json').read_bytes()
assert codec.sha((STAGE / 'parts-control.json').read_bytes()) == reg['parts_control_sha256']
parts.validate(c, mraw)
output = HERE / 'actual'
output.mkdir(exist_ok=False)
children = []
result = dict(status='failed-preserved', POST_attempts=0, first=reg['first'],
              deadline=reg['deadline'], head=reg['head'], strength_success=False)


def guard():
    release.check_clock(c['clock'])
    guardian = common.process_info(reg['guardian']['pid'])
    assert (guardian and guardian['start_ticks'] == reg['guardian']['start_ticks']
            and guardian['state'] not in ['Z', 'X'])
    for path, digest in reg['pins'].items():
        assert common.digest(Path(path)) == digest


def get(args):
    return common.get_with_retries(args, guard, children)


def runs():
    rows = []
    for page in range(1, 11):
        batch = get([f'repos/Eta06/HarbiChess/actions/workflows/{common.WORKFLOW}/runs'
                     f'?event=workflow_dispatch&per_page=10&page={page}'])['workflow_runs']
        assert len(batch) <= 10
        rows.extend(batch)
        if len(batch) < 10:
            return rows
    raise RuntimeError('bounded run inventory incomplete')


try:
    guard()
    assert get(['repos/Eta06/HarbiChess/git/ref/heads/main'])['object']['sha'] == reg['head']
    prior = {r['id'] for r in runs()}
    request = json.dumps(dict(ref='main', inputs=dict(
        parts_control_sha=reg['parts_control_sha256'],
        control_prefix=reg['parts_control_sha256'][:12]))).encode()
    result['POST_attempts'] = 1
    (output / 'single-dispatch-started.json').write_text(json.dumps(result))
    try:
        common.execute_gh(['--method', 'POST',
                           f'repos/Eta06/HarbiChess/actions/workflows/{common.WORKFLOW}/dispatches'],
                          request, min(15, common.END-time.time()), children)
        result['POST_status'] = 'accepted-no-repeat'
    except common.Failure:
        result['POST_status'] = 'ambiguous-reconcile-exact-owned-run-no-repeat'
    ident = None
    while True:
        guard()
        run = (common.select_owned(runs(), prior) if ident is None
               else common.validate_run(get([f'repos/Eta06/HarbiChess/actions/runs/{ident}'])))
        if run:
            if ident is None:
                (output / 'owned-run.json').write_text(json.dumps(run))
            ident = run['id']
            result['owned_run_id'] = ident
            if run['status'] == 'completed':
                assert run['conclusion'] == 'success', 'owned workflow failed, preserve'
                break
        time.sleep(min(10, common.END-time.time()))
    guard()
    public = release.Transport('', c['clock'], dict(id=404068972,
        tag='ufuk-cpu-ownplay-state-20261005',
        target_commitish='31a18e981e27e696fcca69c7e62210e7f0e89830', initial_assets=1))
    inventory = public.inventory()
    for ref in c['protected_assets']:
        matches = [a for a in inventory if a['name'] == ref['name']]
        assert len(matches) == 1 and matches[0]['size'] == ref['bytes']
    values, assets = [], []
    for row in c['parts']:
        guard()
        raw = public.download(row['asset_name'], row['bytes'])
        assert codec.sha(raw) == row['sha256']
        matches = [a for a in inventory if a['name'] == row['asset_name']]
        assert len(matches) == 1
        public.asset(matches[0], row['asset_name'], raw)
        values.append(raw)
        assets.append({k: matches[0][k] for k in ['id', 'name', 'size', 'browser_download_url']})
    joined = b''.join(values)
    assert len(joined) == c['full_capsule_bytes'] and codec.sha(joined) == c['full_capsule_sha256']
    originals = codec.verify(joined, json.loads(mraw))
    assert len(originals) == c['original_files'] == 453
    guard()
    result.update(status='PASS-public-eight-parts-and-all453-original-byte-SHA-not-runtime-strength',
                  parts=assets, originals=originals, capsule_sha256=codec.sha(joined),
                  anonymous_full_parts_and_originals_verified=True, POST_attempts=1,
                  protected_V6_V7_V8_preserved=True)
except BaseException as error:
    result.update(error_type=type(error).__name__,
                  error_code=str(error) if isinstance(error, common.Failure) else 'details-suppressed')
finally:
    result['finished_epoch'] = time.time()
    (output / 'result.json').write_text(json.dumps(result, sort_keys=True))
    print(json.dumps({k: result.get(k) for k in ['status', 'owned_run_id', 'finished_epoch']}),
          flush=True)
