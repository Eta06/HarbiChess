"""Bounded CPU-only phase controller; historical clocks are never rewritten."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    registration = Path(sys.argv[1])
    reg = json.loads(registration.read_bytes())
    assert reg['schema'] == 'UFUK-DEVAM-bounded-cpu-phase-v1'
    assert sha(__file__) == reg['controller_sha256']
    out = Path(reg['output'])
    assert out.is_relative_to(Path('/dev/shm/harbichess-continuation-20261007'))
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {reg['cpu_core']})
    spec = importlib.util.spec_from_file_location('phase_memory_guard', reg['memory_guard_path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    budget = module.CgroupMemoryBudget(15 * 2**30)
    last_pin_check = -float('inf')
    def guard(force=False):
        nonlocal last_pin_check
        now = time.time()
        assert reg['first_epoch'] <= now < reg['deadline_epoch'] <= reg['operator_end_epoch']
        assert shutil.disk_usage('/workspace').free >= 256 * 2**20
        budget.check()
        if force or time.monotonic() - last_pin_check >= 30:
            for path, expected in reg['input_pins'].items():
                assert sha(path) == expected, path
            last_pin_check = time.monotonic()
    child = None
    error = None
    status = 'FAILED-preserved'
    try:
        guard(True)
        environment = {**os.environ, 'PYTHONPATH': reg['checkout'] + '/src',
            'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
            'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''}
        with (out/'stdout.log').open('x') as stdout, (out/'stderr.log').open('x') as stderr:
            child = subprocess.Popen(reg['command'], cwd=reg['checkout'], env=environment,
                stdout=stdout, stderr=stderr, start_new_session=True)
            while child.poll() is None:
                guard()
                time.sleep(.5)
            if child.returncode:
                raise RuntimeError('phase exit ' + str(child.returncode))
        guard(True)
        status = 'PASS-phase-executed-not-strength'
    except BaseException as exc:
        error = repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    result = dict(status=status, error=error, first_epoch=reg['first_epoch'],
        deadline_epoch=reg['deadline_epoch'], operator_end_epoch=reg['operator_end_epoch'],
        finished_epoch=time.time(), registration_sha256=sha(registration),
        controller_sha256=sha(__file__), strength_success_claimed=False)
    (out/'finished.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
