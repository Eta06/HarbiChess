"""Read-only keepalive of the owner supplied host; no CLI allocation fallback."""

import json
import subprocess
import time
from pathlib import Path

END = 1791180000
ROOT = Path('/workspace/work/harbichess/a100')
KEY = '/workspace/attachments/99c5d079-2928-47f7-ae36-ed94a19b2a24/harbichess_colab_id_ed25519'
TRANSPORT = '/workspace/work/cloudflared-a100/cloudflared'
KNOWN = '/workspace/work/cloudflared-a100/known_hosts'


def main():
    while time.time() < END:
        first = time.time()
        row = {'epoch': first, 'allocation_or_reset': False, 'remote_signals': False}
        cmd = ['ssh', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UserKnownHostsFile='+KNOWN, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
               '-o', 'ProxyCommand='+TRANSPORT+' access ssh --hostname %h', 'root@a100.emda.tr',
               "python3 -c 'import pathlib,json,subprocess; print(json.dumps(dict(boot_id="
               "pathlib.Path(\"/proc/sys/kernel/random/boot_id\").read_text().strip(),gpu="
               "subprocess.check_output([\"nvidia-smi\",\"--query-gpu=name,utilization.gpu\","
               "\"--format=csv,noheader\"],text=True).strip())))'"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=min(35, max(.1, END-time.time())))
            row['returncode'] = p.returncode
            if p.returncode == 0:
                row['observed'] = json.loads(p.stdout)
            else:
                row['error'] = ('host-key-mismatch' if 'REMOTE HOST IDENTIFICATION HAS CHANGED'
                                in p.stderr else 'connection-failed-details-suppressed')
        except Exception as exc:
            row['error'] = type(exc).__name__
        row['finished_epoch'] = time.time()
        with (ROOT/'existing-CF-keepalive-v4.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        # Poll intervals never block the parent conversation; this is a background owner.
        while time.time() < min(first+900, END):
            time.sleep(min(30, max(0, min(first+900, END)-time.time())))


if __name__ == '__main__':
    main()
