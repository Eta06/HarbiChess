"""Execute the frozen build/registry/proof plan within the existing ROOT clock."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    args = parser.parse_args()
    commands = json.loads((args.specs / "commands.json").read_bytes())
    if len(commands) != 2 or Path(commands[0][1]).name != "contracts.py":
        raise ValueError("expected frozen contracts and metadata registry plan")
    registry = Path(commands[1][commands[1].index("--output") + 1])
    for command in commands:
        subprocess.run(command, check=True)
    final = json.loads((registry / "commands.json").read_bytes())
    if len(final) != 1 or Path(final[0][1]).name != "prove.py":
        raise ValueError("expected frozen proof or fresh-fit executor")
    subprocess.run(final[0], check=True)


if __name__ == "__main__":
    main()
