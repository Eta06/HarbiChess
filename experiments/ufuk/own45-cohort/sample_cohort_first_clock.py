"""Owner-run once BEFORE config writes; no job launch and no future/reset clock."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    first = time.time()
    # Exclusive publish deliberately cannot replace a previous owned first clock.
    value = {'schema': 'own45-common-original-firstclock-v1',
             'original_training_started_epoch': first, 'slots': [4, 5],
             'scope': 'bothseed/train/audit original clocks include all subsequent sidecar writes',
             'sampler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'pid': os.getpid()}
    with a.output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value))


if __name__ == '__main__':
    main()
