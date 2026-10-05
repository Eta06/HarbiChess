"""Rebuild the versioned CPython extension into a NEW output directory."""

import argparse
import hashlib
import json
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
source = Path(__file__).with_name("forward18_v2.c")
include = Path(sysconfig.get_paths()["include"])
binary = a.output / ("_ownq_forward18_v2" + sysconfig.get_config_var("EXT_SUFFIX"))
cmd = [
    "cc",
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-I" + str(include),
    str(source),
    "-o",
    str(binary),
    "-lm",
]
subprocess.run(cmd, check=True)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(
    command=cmd,
    source_sha256=sha(source),
    binary_sha256=sha(binary),
    python=sys.version,
    platform=sysconfig.get_platform(),
    header_sha256=sha(include / "Python.h"),
    finished_epoch=time.time(),
    training_state_modified=False,
)
(a.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
