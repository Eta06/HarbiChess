"""Fixed local22-byte proof and simulated blob JSON roundtrip; no network/opaque writes."""

import json
import os
import time
import tracemalloc
from pathlib import Path

import blob_delivery_v6 as delivery
import fresh_binding
import launcher_inputs_v6 as launcher
import receiver_v6 as receiver
import state_bundle as bundle

ROOT = Path(__file__).resolve().parent


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    start = time.time()
    tracemalloc.start()
    paths = launcher.geometry()
    originals = {k: p.read_bytes() for k, p in paths.items()}
    capsule, manifest = bundle.encode(originals, launcher.E8.read_bytes())
    del originals
    restored = bundle.decode(capsule, launcher.E8.read_bytes(), manifest)
    for path, raw in restored.items():
        bundle.require(raw == paths[path].read_bytes(), "all22-fixed-original-byte-equality")
    del restored, raw
    manifest.update(
        transport_schema=delivery.TRANSPORT_SCHEMA,
        status="PENDING-ROOT-APPROVAL",
        source_commit=fresh_binding.SOURCE,
        workflow_path=receiver.WORKFLOW,
        workflow_sha256=bundle.sha((ROOT / "fresh-selflearning-state-blob-v6.yml").read_bytes()),
        receiver_sha256=bundle.sha((ROOT / "receiver_v6.py").read_bytes()),
        source_bindings={n: bundle.sha((ROOT / n).read_bytes()) for n in receiver.SUPPORT},
        action_pins={"checkout": "11bd71901bbe5b1630ceea73d27597364c9af683"},
        release=dict(
            id=None, tag="PENDING-ROOT-NEW-RELEASE", target_commitish=None, initial_assets=0
        ),
        original_geometry={k: str(p) for k, p in paths.items()},
        blob=dict(
            sha1=delivery.git_blob_sha(capsule),
            bytes=len(capsule),
            sha256=bundle.sha(capsule),
            repository="Eta06/HarbiChess",
            status="PENDING-ROOT-POST-VERIFY",
        ),
    )
    _, body = delivery.create_blob_request(capsule, manifest)
    # Simulate the documented Git blob response shape, not an actual API success claim.
    expected = dict(manifest)
    expected["blob"] = dict(manifest["blob"], status="ROOT-verified-unreferenced-transient-blob")
    response = dict(
        sha=manifest["blob"]["sha1"],
        size=len(capsule),
        encoding="base64",
        url="https://api.github.com" + delivery.BLOB_API + "/" + manifest["blob"]["sha1"],
        content=json.loads(body)["content"],
    )
    del body
    raw_response = bundle.canonical(response)
    del response
    decoded = delivery.decode_blob_response(raw_response, expected)
    bundle.require(decoded == capsule, "simulated-blob-json-byte-roundtrip")
    peak = tracemalloc.get_traced_memory()[1]
    bundle.require(peak <= 16 * 1024**2, "traced-allocation16MiB")
    (ROOT / "manifest-DRAFT.json").write_bytes(bundle.canonical(manifest) + b"\n")
    proof = dict(
        schema="fresh-state-blob-v6-local-review-v1",
        status="pass",
        elapsed_seconds=time.time() - start,
        all22_original_byte_sha_verified=True,
        simulated_blob_json_exact=True,
        actual_api_calls=0,
        opaque_files_written=0,
        raw_bytes=manifest["raw_bytes"],
        encoded_bytes=len(capsule),
        permanent_assets=1,
        git_blob_sha1=delivery.git_blob_sha(capsule),
        capsule_sha256=bundle.sha(capsule),
        peak_python_traced_allocation_bytes=peak,
        manifest_sha256=bundle.sha((ROOT / "manifest-DRAFT.json").read_bytes()),
    )
    (ROOT / "local-roundtrip.json").write_bytes(bundle.canonical(proof) + b"\n")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
