"""Seal, qualify and run the fixed known8 screen under one original ROOT clock."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    directory = Path(plan["study"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        sys.executable, str(directory / "seal.py"),
        "--draft", plan["draft"], "--draft-sha256", plan["draft_sha256"],
        "--first", str(plan["first"]), "--operator-end", str(plan["operator_end"]),
        "--output", str(directory / "protocol.json"),
    ], check=True)
    subprocess.run([
        sys.executable, str(directory / "qualify_profile.py"),
        "--protocol", str(directory / "protocol.json"),
        "--output", str(output / "profile.json"), "--first-epoch", str(plan["first"]),
    ], check=True)
    subprocess.run([
        sys.executable, str(directory / "run.py"), "--checkout", plan["checkout"],
        "--study", str(directory), "--qualification", str(output / "profile.json"),
        "--output", str(output / "arena"), "--first-epoch", str(plan["first"]),
    ], check=True)


if __name__ == "__main__":
    main()
