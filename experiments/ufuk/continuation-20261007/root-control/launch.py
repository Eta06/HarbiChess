"""Register each new phase once before spawning; never change an old timer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
CORE = Path('/workspace/work/harbichess/cpu-additive-source-6fcc8b4')
PYTHON = '/workspace/HarbiChess/.venv/bin/python'
CONTROLLER = BASE/'controller.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, body):
    with Path(path).open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')

def launch(directory, output, command, pins, core, first, duration, phase):
    operator = json.loads((BASE/'operator-window.json').read_bytes())
    end = first + duration
    assert operator['first_epoch'] <= first <= time.time() < end <= operator['operator_end_epoch']
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    reg = dict(schema='UFUK-DEVAM-bounded-cpu-phase-v1', phase=phase, output=str(output),
        first_epoch=first, deadline_epoch=end, operator_end_epoch=operator['operator_end_epoch'],
        cpu_core=core, checkout=str(CORE), controller_sha256=sha(CONTROLLER),
        memory_guard_path=str(CORE/'src/harbichess/training/cgroup_budget.py'),
        input_pins={str(p):sha(p) for p in pins}, command=list(command),
        source_core_commit='6fcc8b476d25495d1c9c413e55b2c7ba4794013e',
        historical_timer_reset=False, GPU_used=False, hash_guard_interval_seconds=30)
    path = directory/'controller-registration.json'
    write(path, reg)
    with (directory/'controller.log').open('x') as log:
        child = subprocess.Popen([PYTHON, str(CONTROLLER), str(path)], stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    owner = directory/'owner-process.json'
    write(owner, dict(pid=child.pid, start_ticks=ticks, registration=str(path),
        first_epoch=first, deadline_epoch=end, operator_end_epoch=operator['operator_end_epoch']))
    catalog = BASE/'operator-catalog.json'
    body = json.loads(catalog.read_bytes())
    assert str(owner) not in body['owner_receipts']
    body['owner_receipts'].append(str(owner))
    temporary = catalog.with_suffix('.launch.tmp')
    temporary.write_text(json.dumps(body,indent=2)+'\n')
    temporary.replace(catalog)
    return dict(pid=child.pid, registration=str(path), first_epoch=first, deadline_epoch=end)
