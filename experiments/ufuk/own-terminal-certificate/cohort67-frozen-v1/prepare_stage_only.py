"""Emit an immutable staging payload for root; performs no SSH or compute."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--emit-remote-python", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    root = args.inventory.parent
    assets = {}
    for name, digest in inventory["file_sha256"].items():
        if name == "transport-receipt.json" or name.endswith(".txt"):
            continue
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        assets[name] = {"sha256": digest, "base64": base64.b64encode(raw).decode()}
    payload = {"root": inventory["remote_root"], "assets": assets}
    code = """import base64,hashlib,json,os,tempfile
from pathlib import Path
payload=json.loads(PAYLOAD_LITERAL)
root=Path(payload['root']);root.mkdir(parents=True,exist_ok=True)
for name,asset in payload['assets'].items():
 path=root/name;raw=base64.b64decode(asset['base64'])
 assert hashlib.sha256(raw).hexdigest()==asset['sha256']
 if path.exists():
  assert path.read_bytes()==raw
  continue
 fd,tmp=tempfile.mkstemp(prefix='.stage-',dir=root)
 try:
  with os.fdopen(fd,'wb') as stream:
   stream.write(raw);stream.flush();os.fsync(stream.fileno())
  os.link(tmp,path)
 finally:
  Path(tmp).unlink(missing_ok=True)
for name,asset in payload['assets'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==asset['sha256']
receipt={'status':'staged-immutable-cohort-inputs-no-compute','root':str(root)}
receipt['file_sha256']={name:asset['sha256'] for name,asset in payload['assets'].items()}
receipt['training_or_games_launched']=0
print(json.dumps(receipt))
""".replace("PAYLOAD_LITERAL", repr(json.dumps(payload)))
    with args.emit_remote_python.open("x") as stream:
        stream.write(code)
    print(json.dumps({"staging_payload": str(args.emit_remote_python), "no_remote_actions": True}))


if __name__ == "__main__":
    main()
