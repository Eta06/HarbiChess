"""Owner SSH transport for immutable receipts; importing this module performs no network I/O."""

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import shlex
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

SSH_HELPER = Path('/workspace/work/harbichess/a100/ssh_colab_access.py')
REMOTE_CODE = '''import base64,json,os,pathlib,sys,tempfile
mode,path=sys.argv[1:3]
p=pathlib.Path(path)
def read(p):
 if p.is_symlink(): raise ValueError('symlink')
 return p.read_bytes()
def once(p,data):
 if p.exists() or p.is_symlink():
  if read(p)!=data: raise ValueError('existing mismatch')
  return
 fd,name=tempfile.mkstemp(prefix='.mlx-transport-',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
  try: os.link(name,p)
  except FileExistsError:
   if read(p)!=data: raise ValueError('raced mismatch')
  fd=os.open(p.parent,os.O_DIRECTORY)
  try: os.fsync(fd)
  finally: os.close(fd)
 finally: os.unlink(name)
if mode=='read': sys.stdout.buffer.write(read(p))
elif mode=='publish':
 data=sys.stdin.buffer.read();digest=sys.argv[3]
 import hashlib
 if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('transfer hash')
 once(p,data);once(pathlib.Path(path+'.sha256'),(digest+'\\n').encode())
elif mode=='verify':
 print(json.dumps({'receipt':base64.b64encode(read(p)).decode(),
 'checksum':base64.b64encode(read(pathlib.Path(path+'.sha256'))).decode()}))
else: raise ValueError('mode')
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def remote_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('noncanonical remote path')
    if not path.is_relative_to('/content') or path == PurePosixPath('/content'):
        raise ValueError('remote path outside content')
    return value


def request(call, mode, path, *, data=None, sha=None):
    argv = ['python3', '-c', REMOTE_CODE, mode, remote_path(path)]
    if sha is not None:
        argv.append(sha)
    result = call(shlex.join(argv), timeout=45, input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError('owner transport request failed')
    return result.stdout


def read_manifest(call, path):
    data = request(call, 'read', path)
    json.loads(data)  # Inflight/partial JSON cannot be returned as a complete manifest.
    return data


def publish_receipt(call, remote, local, expected):
    remote_path(remote)
    if re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('invalid receipt digest')
    local = Path(local)
    if not local.is_absolute() or local.is_symlink():
        raise ValueError('invalid local receipt path')
    data = local.read_bytes()
    if digest(data) != expected:
        raise ValueError('local receipt hash mismatch')
    json.loads(data)
    request(call, 'publish', remote, data=data, sha=expected)
    packet = json.loads(request(call, 'verify', remote))
    receipt = base64.b64decode(packet['receipt'], validate=True)
    checksum = base64.b64decode(packet['checksum'], validate=True)
    if receipt != data or digest(receipt) != expected or checksum != (expected + '\n').encode():
        raise ValueError('remote readback mismatch')
    if local.read_bytes() != data:
        raise ValueError('local receipt changed during transfer')
    return data, receipt, checksum, {
        'status': 'immutable-published-receipt-and-sha-readback-verified',
        'path': remote, 'sha256': expected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--ssh-helper-sha256', required=True)
    parser.add_argument('mode', choices=['read', 'publish'])
    parser.add_argument('remote')
    parser.add_argument('local', nargs='?')
    parser.add_argument('sha256', nargs='?')
    args = parser.parse_args()
    started = time.time()
    receipt = None
    log = None
    try:
        if digest(SSH_HELPER.read_bytes()) != args.ssh_helper_sha256:
            raise ValueError('SSH helper hash mismatch')
        spec = importlib.util.spec_from_file_location('owner_ssh', SSH_HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args.audit_root.mkdir(parents=True, exist_ok=True)
        log = args.audit_root / uuid.uuid4().hex
        log.mkdir()
        if args.mode == 'read':
            if args.local is not None or args.sha256 is not None:
                raise ValueError('unexpected read arguments')
            data = read_manifest(module.call, args.remote)
            (log / 'manifest.json').write_bytes(data)
            receipt = data
        else:
            data, readback, checksum, result = publish_receipt(
                module.call, args.remote, args.local, args.sha256
            )
            (log / 'local-receipt.json').write_bytes(data)
            (log / 'remote-readback.json').write_bytes(readback)
            (log / 'remote-readback.sha256').write_bytes(checksum)
            receipt = (json.dumps(result) + '\n').encode()
        (log / 'owner-command-clock.json').write_text(json.dumps({
            'mode': args.mode, 'remote': args.remote, 'started_epoch': started,
            'finished_epoch': time.time(), 'status': 'verified',
            'payload_sha256': digest(data),
        }) + '\n')
    except Exception:
        if log is not None:
            (log / 'owner-command-clock.json').write_text(json.dumps({
                'mode': args.mode, 'remote': args.remote, 'started_epoch': started,
                'finished_epoch': time.time(), 'status': 'failed-no-publication-claim',
            }) + '\n')
        # Never expose raw SSH/auth errors, stderr, commands, or environment.
        print('owner transport failed', file=sys.stderr)
        return 1
    sys.stdout.buffer.write(receipt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
