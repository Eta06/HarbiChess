"""ROOT metadata builder; clocks remain NULL, no eligibility/selection/model/game calls."""

import argparse
import json
from pathlib import Path

from bindings import SEEDS, read, sha
from select_books import publish


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("known-protocol", "known-profile", "known-audit", "known-six", "audit-set",
                 "known-selection-inventory", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    known = json.loads(a.known_protocol.read_bytes())
    helpers = {x.name: sha(x) for x in Path(__file__).parent.glob("*.py")}
    eligibility = dict(schema="ONE-fixed-candidate-eligibility-spec-v2", first=None, deadline=None,
                       operator_end_epoch=known["ROOToperator_end_epoch"], cpu_core=None,
                       known_protocol=ref(a.known_protocol), known48_profile=ref(a.known_profile),
                       known160_audit=ref(a.known_audit), known_six_replay=ref(a.known_six),
                       collection_audit_set=ref(a.audit_set), helper_sha256=helpers)
    inventory = json.loads(a.known_selection_inventory.read_bytes())
    if inventory["status"] != "PASS-complete-recorded-known-selection-inventory":
        raise ValueError("ROOT independently reconciled ALL-known selection inventory required")
    manifest = dict(schema="ONE-current-selected-DAG-exposure-manifest-v2",
                    classical_parent_lineages={}, known_books=inventory["known_books"],
                    known_arenas=inventory["known_arenas"],
                    additional_recorded_alias_bins=inventory.get("alias_bins", []),
                    additional_fullhistory_rows=inventory.get("fullhistory_rows", []),
                    all_known_selection_inventory_complete=True,
                    known_selection_inventory=ref(a.known_selection_inventory))
    for seed in SEEDS:
        contract = read(known["teachers"][str(seed)]["contract"])
        provenance = read(dict(path=contract["target_provenance_path"],
                               sha256=contract["target_provenance_sha256"]))
        manifest["classical_parent_lineages"][str(seed)] = provenance[
            "selection_receipt"]["original_lineage"]
    a.output.mkdir(exist_ok=False)
    publish(a.output / "eligibility-spec-DRAFT.json", eligibility)
    publish(a.output / "exposure-manifest.json", manifest)
    cache = Path("/dev/shm/harbichess-confirmation-source-readiness-20261005")
    selection = dict(schema="ONE-after-eligibility-blind-book-selection-registration-v2",
                     status="DRAFT-not-registered-no-draw", first=None, deadline=None,
                     operator_end_epoch=known["ROOToperator_end_epoch"], cpu_core=None,
                     eligibility=dict(path="ROOT-FILL-ELIGIBILITY", sha256="ROOT-FILL"),
                     known_protocol=ref(a.known_protocol),
                     exposure_manifest=ref(a.output / "exposure-manifest.json"),
                     human_pgn=ref(cache / "lichess-2024-10-prefix-decoded.pgn"),
                     compressed_source=ref(cache / "lichess-2024-10-prefix-2MiB.pgn.zst"),
                     public_source_binding=ref(cache / "source-binding.json"),
                     decode_binding=ref(cache / "decode-binding.json"), helper_sha256=helpers)
    publish(a.output / "book-selection-spec-DRAFT.json", selection)


if __name__ == "__main__":
    main()
