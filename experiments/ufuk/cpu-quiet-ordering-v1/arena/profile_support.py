"""Publish-once profile receipts; successful work includes original final clock."""

import json
import os
import time
from pathlib import Path


def publish_result(path, result, deadline, clock=None):
    clock = time.time if clock is None else clock
    receipt = dict(result)
    receipt["rows"] = list(result["rows"])
    finished = clock()
    receipt["finished_epoch"] = finished
    if receipt["status"].startswith("PASS") and finished >= deadline:
        receipt["status"] = "failed-preserved"
        receipt["rows"].append({"error": "original600 profile deadline includes final audit"})
    data = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return receipt
