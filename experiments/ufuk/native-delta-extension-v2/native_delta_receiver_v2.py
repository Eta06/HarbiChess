"""Fixed Release receiver proposal. Network execution only under reviewed Actions guard.

Imports the standard-library decoder. No Torch, shell commands, arbitrary URLs,
asset overwrite/deletion, repository mutation or allocation APIs.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import native_delta_binding_v2 as codec

ROOT = Path(__file__).resolve().parent
REPO = "Eta06/HarbiChess"
RELEASE = 401698693
TAG = "port-linux-preflight-20261002"
API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"
END = 1791273600  # 2026-10-06T08:00:00Z; transport cannot extend this operator cap.
ASSET_CAP = 200
BYTE_CAP = 8 * 1024**3
CONTROL = ROOT / "transport-control-PENDING.json"
ALLOWLIST = {
    "8d9929a1b1c9e862194c2024c6b2a391ea6bc00158b3aa7c647ee19aef3b60ec": (
        "20262605-shrunk/manifest.json"
    ),
    "10d2ad259754ec1ea1bdca5e4affb41bb5af06b259c03af26137f2d1a0b37806": (
        "20262606-shrunk/manifest.json"
    ),
    "bbff8b5f8132ae88fbb00a40e0b943dff4ae2f446fef877909d8db295cc201f8": (
        "20262705-residual/manifest.json"
    ),
    "ce72e2112ed978c0648ac4e07459d13ebd16226ffeb0ca028b210373dc2e17d0": (
        "20262706-residual/manifest.json"
    ),
}
CODEC_SHA = "878868940b5bae15c182dce55215320b5f0ba3692b9c5eac26988204832a31d6"
BINDING_ADAPTER_SHA = "c18f507afeca44920fddbf4a9fcf89242f06bd35a0c0e12ed921aecf85769199"
PARENT_NAME = "sha256-" + codec.PUBLIC_PARENT_ARCHIVE_SHA256 + ".tar.gz"
SAFE_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}


class Failure(RuntimeError):
    """Errors are fixed codes only; never remote bodies, token or connection URLs."""


def check_clock(control, now=None):
    now = time.time() if now is None else now
    if (
        control.get("schema") != "cpu-native-delta-transport-control-v2"
        or control.get("status") != "ROOT-approved-frozen-transport"
        or control.get("operator_end_epoch") != END
    ):
        raise Failure("transport-not-approved")
    start, end = control["started_epoch"], control["deadline_epoch"]
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not start <= now < end <= min(start + 1800, END)
    ):
        raise Failure("original-transport-clock-invalid-or-expired")
    return min(60, end - now)


def load_manifest(sha):
    if sha not in ALLOWLIST:
        raise Failure("manifest-not-fixed-allowlist")
    data = (ROOT / ALLOWLIST[sha]).read_bytes()
    if (
        codec.digest(data) != sha
        or codec.digest((ROOT / "native_delta_codec.py").read_bytes()) != CODEC_SHA
    ):
        raise Failure("manifest-or-codec-source-changed")
    value = json.loads(data)
    if (
        value["codec_sha256"] != CODEC_SHA
        or value["binding_adapter_sha256"] != BINDING_ADAPTER_SHA
        or value["binding_adapter_version"] != "fixed-shrunk-residual-v2"
        or codec.digest((ROOT / "native_delta_binding_v2.py").read_bytes()) != BINDING_ADAPTER_SHA
    ):
        raise Failure("manifest-codec-binding")
    return value


def chunk_name(manifest, ordinal):
    return f"cpu-native-delta-{manifest['packet_sha256']}-chunk-{ordinal:03d}.base64"


def check_asset(asset, name, size, sha):
    ident = asset.get("id")
    if (
        type(ident) is not int
        or ident <= 0
        or asset.get("name") != name
        or asset.get("size") != size
        or asset.get("state") != "uploaded"
        or asset.get("url") != f"{API}/repos/{REPO}/releases/assets/{ident}"
        or asset.get("browser_download_url") != public_url(name)
        or asset.get("digest") not in (None, "sha256:" + sha)
    ):
        raise Failure("release-asset-metadata-identity-size-digest")


def public_url(name):
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._" for c in name
    ):
        raise Failure("unsafe-asset-name")
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if (
            parsed.scheme != "https"
            or parsed.netloc != parsed.hostname
            or parsed.hostname not in SAFE_HOSTS
            or parsed.fragment
        ):
            raise Failure("anonymous-download-redirect-not-github")
        # Fresh GET contains no Authorization and inherits configured proxy/CA trust.
        return urllib.request.Request(
            newurl, headers={"User-Agent": "HarbiChess-native-delta/1"}, method="GET"
        )


class Transport:
    def __init__(self, token, control):
        self.token, self.control = token, control
        self.api_opener = urllib.request.build_opener(NoRedirect())
        self.public_opener = urllib.request.build_opener(PublicRedirect())

    def request(self, url, *, data=None, public=False):
        timeout = check_clock(self.control)
        headers = {"User-Agent": "HarbiChess-native-delta/1"}
        if not public:
            headers.update(
                {
                    "Authorization": "Bearer " + self.token,
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                }
            )
        if data is not None:
            headers.update(
                {"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}
            )
        req = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data is not None else "GET"
        )
        try:
            opener = self.public_opener if public else self.api_opener
            return opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise Failure(f"http-status-{error.code}-details-suppressed") from None
        except Failure:
            raise
        except Exception:
            raise Failure("network-failure-details-suppressed") from None

    def api_json(self, path, data=None):
        origin = UPLOAD if data is not None else API
        if path != f"/repos/{REPO}/releases/{RELEASE}" and not path.startswith(
            f"/repos/{REPO}/releases/{RELEASE}/assets?"
        ):
            raise Failure("fixed-api-path-required")
        with self.request(origin + path, data=data) as response:
            raw = response.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise Failure("api-json-size-bound")
        return json.loads(raw)

    def inventory(self):
        release = self.api_json(f"/repos/{REPO}/releases/{RELEASE}")
        if (
            release.get("id") != RELEASE
            or release.get("tag_name") != TAG
            or release.get("draft") is not False
            or release.get("upload_url")
            != f"{UPLOAD}/repos/{REPO}/releases/{RELEASE}/assets{{?name,label}}"
        ):
            raise Failure("existing-public-release-identity")
        rows, ids, names = [], set(), set()
        for page in range(1, 11):
            batch = self.api_json(
                f"/repos/{REPO}/releases/{RELEASE}/assets?per_page=100&page={page}"
            )
            if not isinstance(batch, list) or len(batch) > 100:
                raise Failure("bounded-canonical-release-pagination")
            for row in batch:
                if (
                    row["id"] in ids
                    or row["name"] in names
                    or type(row["size"]) is not int
                    or row["size"] < 0
                ):
                    raise Failure("duplicate-or-invalid-release-inventory")
                ids.add(row["id"])
                names.add(row["name"])
            rows.extend(batch)
            if len(batch) < 100:
                if len(rows) > ASSET_CAP or sum(row["size"] for row in rows) > BYTE_CAP:
                    raise Failure("existing-release-caps")
                return rows
        raise Failure("release-pagination-incomplete-at-bound")

    def upload(self, name, data):
        query = urllib.parse.urlencode({"name": name, "label": name})
        return self.api_json(f"/repos/{REPO}/releases/{RELEASE}/assets?{query}", data=data)

    def download(self, name, size):
        with self.request(public_url(name), public=True) as response:
            raw = response.read(size + 1)
        if len(raw) != size:
            raise Failure("anonymous-full-body-size")
        return raw

    def parent(self):
        """Stream whole public archive/hash; keep only one verified regular E8 member."""
        inventory = self.inventory()
        found = [row for row in inventory if row["name"] == PARENT_NAME]
        if len(found) != 1:
            raise Failure("parent-asset-missing-or-duplicate")
        check_asset(
            found[0],
            PARENT_NAME,
            codec.PUBLIC_PARENT_ARCHIVE_BYTES,
            codec.PUBLIC_PARENT_ARCHIVE_SHA256,
        )
        with self.request(public_url(PARENT_NAME), public=True) as response:
            reader = HashReader(response, self.control)
            selected = None
            total_uncompressed, member_count = 0, 0
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    check_clock(self.control)
                    total_uncompressed += member.size
                    member_count += 1
                    if (
                        not member.isreg()
                        or member.size > 512 * 1024**2
                        or total_uncompressed > 1024**3
                        or member_count > 500
                    ):
                        raise Failure("parent-archive-nonregular-or-size-bound")
                    if member.name == codec.PUBLIC_PARENT_MEMBER:
                        if selected is not None or member.size != 294764:
                            raise Failure("parent-member-duplicate-size")
                        selected = archive.extractfile(member).read()
            while reader.read(1024**2):
                pass
            if (
                reader.bytes != codec.PUBLIC_PARENT_ARCHIVE_BYTES
                or reader.hash.hexdigest() != codec.PUBLIC_PARENT_ARCHIVE_SHA256
            ):
                raise Failure("entire-parent-archive-byte-sha")
        if selected is None or codec.digest(selected) != codec.E8_SHA256:
            raise Failure("exact-public-e8-member-sha")
        return selected


class HashReader:
    def __init__(self, stream, control):
        self.stream, self.control = stream, control
        self.bytes, self.hash = 0, hashlib.sha256()

    def read(self, count):
        check_clock(self.control)
        data = self.stream.read(count)
        self.bytes += len(data)
        if self.bytes > codec.PUBLIC_PARENT_ARCHIVE_BYTES:
            raise Failure("parent-archive-size-bound")
        self.hash.update(data)
        return data


def publish_verified(transport, name, data):
    sha = codec.digest(data)
    inventory = transport.inventory()
    found = [row for row in inventory if row["name"] == name]
    if len(found) > 1:
        raise Failure("duplicate-named-asset")
    if found:
        asset = found[0]
        check_asset(asset, name, len(data), sha)
        disposition = "matching-retry-no-write"
    else:
        if (
            len(inventory) + 1 > ASSET_CAP
            or sum(row["size"] for row in inventory) + len(data) > BYTE_CAP
        ):
            raise Failure("unchanged-public-aggregate-caps")
        asset = transport.upload(name, data)
        check_asset(asset, name, len(data), sha)
        after = [row for row in transport.inventory() if row["name"] == name]
        if len(after) != 1 or after[0]["id"] != asset["id"]:
            raise Failure("uploaded-asset-not-uniquely-published")
        check_asset(after[0], name, len(data), sha)
        disposition = "new-upload"
    if codec.digest(transport.download(name, len(data))) != sha:
        raise Failure("anonymous-full-body-sha")
    return {
        "name": name,
        "bytes": len(data),
        "sha256": sha,
        "asset_id": asset["id"],
        "disposition": disposition,
        "anonymous_full_body_sha_verified": True,
    }


def deterministic_capsule(originals):
    raw = io.BytesIO()
    with (
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name in codec.FILES:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(originals[name]), 0o600, 0
            archive.addfile(info, io.BytesIO(originals[name]))
    return raw.getvalue()


def verify_capsule(raw, expected):
    seen = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as archive:
        for member in archive:
            if not member.isreg() or member.name not in expected or member.name in seen:
                raise Failure("capsule-extra-link-or-duplicate")
            if member.size != expected[member.name]["bytes"]:
                raise Failure("capsule-original-file-size")
            data = archive.extractfile(member).read()
            if codec.digest(data) != expected[member.name]["sha256"]:
                raise Failure("capsule-original-file-sha")
            seen.append(member.name)
    if tuple(seen) != codec.FILES:
        raise Failure("capsule-exact-three-original-files-required")


def receive_chunk(manifest, inputs, transport):
    if inputs["packet_sha256"] != manifest["packet_sha256"]:
        raise Failure("chunk-packet-binding")
    try:
        ordinal, count = int(inputs["ordinal"]), int(inputs["count"])
    except (TypeError, ValueError):
        raise Failure("chunk-integer-inputs") from None
    if str(ordinal) != inputs["ordinal"] or str(count) != inputs["count"]:
        raise Failure("canonical-chunk-integer-inputs")
    if count != len(manifest["chunks"]) or not 0 <= ordinal < count:
        raise Failure("chunk-ordinal-count")
    text = inputs["payload"]
    expected = manifest["chunks"][ordinal]
    try:
        data = text.encode("ascii")
    except (AttributeError, UnicodeEncodeError):
        raise Failure("chunk-ascii-required") from None
    if (
        len(data) != expected["characters"]
        or len(data) > codec.CHUNK_CHARS
        or codec.digest(data) != expected["sha256"]
        or inputs["chunk_sha256"] != expected["sha256"]
    ):
        raise Failure("chunk-exact-sealed-length-sha")
    return publish_verified(transport, chunk_name(manifest, ordinal), data)


def aggregate(manifest, transport):
    inventory = transport.inventory()
    chunks = []
    for item in manifest["chunks"]:
        name = chunk_name(manifest, item["ordinal"])
        found = [row for row in inventory if row["name"] == name]
        if len(found) != 1:
            raise Failure("aggregate-frozen-chunk-missing-or-duplicate")
        check_asset(found[0], name, item["characters"], item["sha256"])
        data = transport.download(name, item["characters"])
        if codec.digest(data) != item["sha256"]:
            raise Failure("aggregate-chunk-full-sha")
        chunks.append(data.decode("ascii"))
    packet = codec.reassemble(manifest, chunks)
    originals = codec.decode(
        packet, transport.parent(), manifest["packet_sha256"], manifest["files"]
    )
    capsule = deterministic_capsule(originals)
    verify_capsule(capsule, manifest["files"])
    name = f"cpu-native-exact-{manifest['packet_sha256']}-{codec.digest(capsule)}.tar.gz"
    receipt = publish_verified(transport, name, capsule)
    verify_capsule(transport.download(name, len(capsule)), manifest["files"])
    return {
        "schema": "cpu-native-delta-public-capsule-proof-v2",
        "status": "public-exact-bytes-pass",
        "packet_sha256": manifest["packet_sha256"],
        "capsule": receipt,
        "native_files": manifest["files"],
        "binding": manifest["binding"],
        "native_strict_runtime_load_executed": False,
        "no_training_or_cuda": True,
    }


def main():
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPO
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_WORKFLOW_REF")
        != "Eta06/HarbiChess/.github/workflows/cpu-native-delta-release-v2.yml@refs/heads/main"
    ):
        raise Failure("fixed-authorized-actions-only")
    control = json.loads(CONTROL.read_text())
    check_clock(control)
    if (
        codec.digest(Path(__file__).read_bytes()) != control["receiver_sha256"]
        or control["codec_sha256"] != CODEC_SHA
        or control["binding_adapter_sha256"] != BINDING_ADAPTER_SHA
        or control["manifest_sha256"] != sorted(ALLOWLIST)
        or codec.digest(
            (ROOT.parents[2] / ".github/workflows/cpu-native-delta-release-v2.yml").read_bytes()
        )
        != control["workflow_sha256"]
    ):
        raise Failure("approved-transport-code-manifest-inventory")
    mode = os.environ.get("CPU_MODE")
    manifest_sha = os.environ.get("CPU_MANIFEST_SHA256")
    manifest = load_manifest(manifest_sha)
    if os.environ.get("CPU_PACKET_SHA256") != manifest["packet_sha256"]:
        raise Failure("dispatch-frozen-packet-sha")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise Failure("configured-actions-token-unavailable")
    transport = Transport(token, control)
    if mode == "chunk":
        result = receive_chunk(
            manifest,
            {
                "packet_sha256": os.environ["CPU_PACKET_SHA256"],
                "ordinal": os.environ.get("CPU_ORDINAL"),
                "count": os.environ.get("CPU_COUNT"),
                "chunk_sha256": os.environ.get("CPU_CHUNK_SHA256"),
                "payload": os.environ.get("CPU_PAYLOAD_BASE64"),
            },
            transport,
        )
    elif mode == "aggregate":
        if any(
            os.environ.get(key)
            for key in ("CPU_ORDINAL", "CPU_COUNT", "CPU_CHUNK_SHA256", "CPU_PAYLOAD_BASE64")
        ):
            raise Failure("aggregate-forbids-chunk-inputs")
        result = aggregate(manifest, transport)
    else:
        raise Failure("fixed-chunk-or-aggregate-mode")
    check_clock(control)
    result["original_transport_deadline_epoch"] = control["deadline_epoch"]
    result["finished_epoch"] = time.time()
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Failure as error:
        raise SystemExit(str(error)) from None
    except Exception:
        raise SystemExit("receiver-failed-details-suppressed") from None
