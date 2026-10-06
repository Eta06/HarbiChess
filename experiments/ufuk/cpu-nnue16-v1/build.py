"""Explicit local C build, synthetic qualification only; no network or search."""

import hashlib
import json
import subprocess
import sysconfig
from pathlib import Path

HERE = Path(__file__).parent


def main():
    output = HERE / ("_kingbucket16" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        "cc",
        "-O3",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        "-std=c11",
        "-I" + sysconfig.get_path("include"),
        str(HERE / "forward.c"),
        "-o",
        str(output),
        "-lm",
    ]
    if output.exists():
        raise FileExistsError("sealed binary already exists; no overwrite")
    subprocess.run(command, check=True)
    receipt = {
        "command": command,
        "source_sha256": hashlib.sha256((HERE / "forward.c").read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "build only; not real-board parity",
    }
    with (HERE / "build-receipt.json").open("x") as f:
        json.dump(receipt, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
