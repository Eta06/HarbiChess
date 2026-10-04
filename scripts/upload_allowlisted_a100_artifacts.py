"""Transfer fixed-manifest public files to one existing GitHub release safely."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

REPOSITORY = "Eta06/HarbiChess"
RELEASE_ID = 401698693
RELEASE_TAG = "port-linux-preflight-20261002"
MANIFEST = Path("docs/runs/UFUK-A100-release-transport-manifest-20261004.json")
API_ROOT = "https://api.github.com"
UPLOAD_HOST = "uploads.github.com"
USER_AGENT = "HarbiChess-allowlisted-release-transfer/1.0"
API_VERSION = "2022-11-28"
BLOCK_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 1024**3
MAX_TOTAL_BYTES = 8 * 1024**3
MAX_ASSETS = 200
MAX_RELEASE_ASSET_PAGES = 10
ASSETS_PER_PAGE = 100
MAX_API_RESPONSE_BYTES = 1024 * 1024
SAFE_DOWNLOAD_HOSTS = {
    "api.github.com",
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


class DeliveryError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Source files and credentialed GitHub API requests may not redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PublicGitHubRedirect(urllib.request.HTTPRedirectHandler):
    """Only anonymous release readback may redirect; create a clean request each hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        try:
            ports_ok = old.port is None and new.port is None
        except ValueError:
            ports_ok = False
        if (
            old.scheme != "https"
            or new.scheme != "https"
            or old.hostname not in SAFE_DOWNLOAD_HOSTS
            or new.hostname not in SAFE_DOWNLOAD_HOSTS
            or not ports_ok
            or new.netloc != new.hostname
            or new.username is not None
            or new.password is not None
        ):
            raise urllib.error.HTTPError(
                newurl, code, "Rejected non-GitHub asset redirect", headers, fp
            )
        return urllib.request.Request(newurl, headers={"User-Agent": USER_AGENT}, method="GET")


NO_REDIRECT = urllib.request.build_opener(RejectRedirect())
PUBLIC_REDIRECT = urllib.request.build_opener(PublicGitHubRedirect())


def _api_request(path: str, token: str, *, include_headers: bool = False):
    parsed = urllib.parse.urlsplit(path)
    allowed_paths = {
        f"/repos/{REPOSITORY}/releases/{RELEASE_ID}",
        f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets",
    }
    if parsed.path.startswith(f"/repos/{REPOSITORY}/releases/assets/"):
        allowed_paths.add(parsed.path)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.fragment
        or parsed.path not in allowed_paths
        or (
            parsed.path.startswith("/repos/Eta06/HarbiChess/releases/assets/")
            and not re.fullmatch(
                rf"/repos/{re.escape(REPOSITORY)}/releases/assets/[1-9][0-9]*", parsed.path
            )
        )
    ):
        raise DeliveryError("api-path-not-allowlisted")
    if parsed.path.endswith("/assets"):
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if (
            set(query) != {"per_page", "page"}
            or len(query["per_page"]) != 1
            or query["per_page"][0] != str(ASSETS_PER_PAGE)
            or len(query["page"]) != 1
            or not query["page"][0].isdigit()
            or not 1 <= int(query["page"][0]) <= MAX_RELEASE_ASSET_PAGES
        ):
            raise DeliveryError("api-assets-pagination-not-allowlisted")
    elif parsed.query:
        raise DeliveryError("api-query-not-allowlisted")
    req = urllib.request.Request(
        API_ROOT + urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, "")),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        },
        method="GET",
    )
    try:
        response = NO_REDIRECT.open(req, timeout=30)
        with response:
            raw = response.read(MAX_API_RESPONSE_BYTES + 1)
            headers = response.headers
    except urllib.error.HTTPError as exc:
        raise DeliveryError(f"github-api-http-{exc.code}") from None
    except Exception:
        raise DeliveryError("github-api-network-error-details-suppressed") from None
    if len(raw) > MAX_API_RESPONSE_BYTES:
        raise DeliveryError("github-api-response-over-limit")
    try:
        result = json.loads(raw)
    except Exception:
        raise DeliveryError("github-api-json-invalid") from None
    return (result, headers) if include_headers else result


def _asset_list(token: str) -> list[dict]:
    all_assets: list[dict] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for page in range(1, MAX_RELEASE_ASSET_PAGES + 1):
        path = (
            f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets?"
            f"per_page={ASSETS_PER_PAGE}&page={page}"
        )
        page_assets, headers = _api_request(path, token, include_headers=True)
        if not isinstance(page_assets, list) or len(page_assets) > ASSETS_PER_PAGE:
            raise DeliveryError("release-assets-page-not-a-bounded-list")
        for asset in page_assets:
            if (
                not isinstance(asset, dict)
                or type(asset.get("id")) is not int
                or asset["id"] <= 0
                or not isinstance(asset.get("name"), str)
                or not asset["name"]
            ):
                raise DeliveryError("release-asset-record-invalid")
            if asset["id"] in seen_ids or asset["name"] in seen_names:
                raise DeliveryError("release-asset-id-or-name-duplicated-across-pages")
            seen_ids.add(asset["id"])
            seen_names.add(asset["name"])
        all_assets.extend(page_assets)
        next_page = _next_asset_page(headers.get("Link"), page)
        if next_page is not None and next_page != page + 1:
            raise DeliveryError("release-assets-next-page-not-sequential")
        if len(page_assets) < ASSETS_PER_PAGE:
            if next_page is not None:
                raise DeliveryError("release-assets-pagination-inconsistent-or-limit-reached")
            return all_assets
        # A full final page may omit Link. Probe only our canonical next page;
        # never follow a response URL or declare a full bounded listing complete.
        if page == MAX_RELEASE_ASSET_PAGES:
            raise DeliveryError("release-assets-pagination-inconsistent-or-limit-reached")
    raise DeliveryError("release-assets-pagination-limit-reached")


def _next_asset_page(link_header: str | None, current_page: int) -> int | None:
    if not link_header:
        return None
    next_pages = []
    for segment in link_header.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="([a-z]+)"\s*', segment)
        if not match:
            raise DeliveryError("release-assets-link-header-invalid")
        if match.group(2) != "next":
            continue
        target = urllib.parse.urlsplit(match.group(1))
        expected_path = f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets"
        try:
            target_query = urllib.parse.parse_qs(
                target.query, keep_blank_values=True, strict_parsing=True
            )
        except ValueError:
            raise DeliveryError("release-assets-next-link-invalid") from None
        if (
            target.scheme != "https"
            or target.netloc != "api.github.com"
            or target.path != expected_path
            or target.fragment
            or set(target_query) != {"per_page", "page"}
            or target_query.get("per_page") != [str(ASSETS_PER_PAGE)]
            or len(target_query.get("page", [])) != 1
            or not target_query["page"][0].isdigit()
        ):
            raise DeliveryError("release-assets-next-link-not-canonical")
        next_pages.append(int(target_query["page"][0]))
    if len(next_pages) > 1:
        raise DeliveryError("release-assets-next-link-duplicated")
    return next_pages[0] if next_pages else None


def _validate_manifest(token: str) -> tuple[list[dict], str]:
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
    ):
        raise DeliveryError("not-canonical-main-manual-dispatch")
    try:
        raw = MANIFEST.read_bytes()
    except Exception:
        raise DeliveryError("fixed-committed-manifest-unavailable") from None
    manifest_sha = hashlib.sha256(raw).hexdigest()
    try:
        obj = json.loads(raw)
    except Exception:
        raise DeliveryError("manifest-json-invalid") from None
    if (
        not isinstance(obj, dict)
        or obj.get("schema") != "harbichess-a100-release-transport-v1"
        or obj.get("release_id") != RELEASE_ID
        or obj.get("release_tag") != RELEASE_TAG
    ):
        raise DeliveryError("manifest-schema-or-fixed-release-mismatch")
    assets = obj.get("assets")
    if not isinstance(assets, list) or not 1 <= len(assets) <= MAX_ASSETS:
        raise DeliveryError("manifest-asset-count-out-of-range")
    names: set[str] = set()
    total = 0
    for item in assets:
        if not isinstance(item, dict) or set(item) != {"name", "bytes", "sha256", "url"}:
            raise DeliveryError("manifest-entry-keys-invalid")
        name = item["name"]
        size = item["bytes"]
        digest = item["sha256"]
        url = item["url"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,180}", name)
            or name in {".", ".."}
            or name in names
        ):
            raise DeliveryError("manifest-asset-name-invalid-or-duplicate")
        if type(size) is not int or not 0 < size <= MAX_ASSET_BYTES:
            raise DeliveryError("manifest-asset-size-out-of-range")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DeliveryError("manifest-sha256-invalid")
        if not isinstance(url, str):
            raise DeliveryError("manifest-source-url-invalid")
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.netloc != host
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com", host)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != f"/{name}"
            or not re.fullmatch(r"/[A-Za-z0-9._/-]+", parsed.path)
            or any(part in {"", ".", ".."} for part in PurePosixPath(parsed.path).parts[1:])
            or PurePosixPath(parsed.path).name != name
        ):
            raise DeliveryError("manifest-source-url-not-fixed-cloudflare-file-url")
        names.add(name)
        total += size
    if total > MAX_TOTAL_BYTES:
        raise DeliveryError("manifest-total-size-exceeds-8gib")

    release = _api_request(f"/repos/{REPOSITORY}/releases/{RELEASE_ID}", token)
    upload_url = (
        f"https://{UPLOAD_HOST}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets{{?name,label}}"
    )
    assets_url = f"{API_ROOT}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets"
    if (
        release.get("id") != RELEASE_ID
        or release.get("tag_name") != RELEASE_TAG
        or release.get("draft") is not False
        or release.get("upload_url") != upload_url
        or release.get("assets_url") != assets_url
    ):
        raise DeliveryError("github-release-identity-mismatch")
    return assets, manifest_sha


def _download_source(asset: dict, destination: Path) -> None:
    request = urllib.request.Request(
        asset["url"],
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    try:
        response = NO_REDIRECT.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        raise DeliveryError(f"source-http-{exc.code}-redirects-disallowed") from None
    except Exception:
        raise DeliveryError("source-network-error-details-suppressed") from None
    expected_size = asset["bytes"]
    expected_digest = asset["sha256"]
    try:
        with response, destination.open("xb") as output:
            if response.status != 200:
                raise DeliveryError("source-http-status-not-200")
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding not in {"", "identity"}:
                raise DeliveryError("source-content-encoding-not-identity")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and content_length != str(expected_size):
                raise DeliveryError("source-content-length-differs-from-manifest")
            received = 0
            digest = hashlib.sha256()
            while True:
                try:
                    block = response.read(min(BLOCK_BYTES, expected_size - received + 1))
                except Exception:
                    raise DeliveryError("source-read-error-details-suppressed") from None
                if not block:
                    break
                received += len(block)
                if received > expected_size:
                    raise DeliveryError("source-body-exceeds-manifest-size")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except DeliveryError:
        raise
    except Exception:
        raise DeliveryError("source-spool-error-details-suppressed") from None
    if received != expected_size or digest.hexdigest() != expected_digest:
        raise DeliveryError("source-size-or-sha256-mismatch")


def _upload_file(asset: dict, path: Path, token: str) -> dict:
    query = urllib.parse.urlencode({"name": asset["name"], "label": asset["name"]})
    target = f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets?{query}"
    connection = http.client.HTTPSConnection(UPLOAD_HOST, timeout=60)
    try:
        connection.putrequest("POST", target, skip_accept_encoding=True)
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("Accept", "application/vnd.github+json")
        connection.putheader("X-GitHub-Api-Version", API_VERSION)
        connection.putheader("User-Agent", USER_AGENT)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(asset["bytes"]))
        connection.endheaders()
        with path.open("rb") as source:
            while True:
                block = source.read(BLOCK_BYTES)
                if not block:
                    break
                connection.send(block)
        response = connection.getresponse()
        raw = response.read(MAX_API_RESPONSE_BYTES + 1)
        if response.status not in (200, 201):
            raise DeliveryError(f"github-upload-http-{response.status}")
        if len(raw) > MAX_API_RESPONSE_BYTES:
            raise DeliveryError("github-upload-response-over-limit")
        try:
            result = json.loads(raw)
        except Exception:
            raise DeliveryError("github-upload-response-json-invalid") from None
        if (
            result.get("name") != asset["name"]
            or result.get("size") != asset["bytes"]
            or type(result.get("id")) is not int
        ):
            raise DeliveryError("github-upload-response-identity-mismatch")
        return result
    except DeliveryError:
        raise
    except Exception:
        raise DeliveryError("github-upload-network-error-details-suppressed") from None
    finally:
        connection.close()


def _public_asset_readback(asset: dict, metadata: dict) -> str:
    asset_id = metadata.get("id")
    expected_api_url = f"{API_ROOT}/repos/{REPOSITORY}/releases/assets/{asset_id}"
    if (
        type(asset_id) is not int
        or metadata.get("name") != asset["name"]
        or metadata.get("size") != asset["bytes"]
        or metadata.get("url") != expected_api_url
    ):
        raise DeliveryError("github-asset-metadata-identity-mismatch")
    digest_field = metadata.get("digest")
    expected_digest_field = f"sha256:{asset['sha256']}"
    if digest_field is not None and digest_field != expected_digest_field:
        raise DeliveryError("github-asset-api-digest-mismatch")
    req = urllib.request.Request(
        expected_api_url,
        headers={"Accept": "application/octet-stream", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        response = PUBLIC_REDIRECT.open(req, timeout=30)
    except urllib.error.HTTPError as exc:
        raise DeliveryError(f"anonymous-asset-readback-http-{exc.code}") from None
    except Exception:
        raise DeliveryError("anonymous-asset-readback-network-error-details-suppressed") from None
    size = 0
    digest = hashlib.sha256()
    try:
        with response:
            if response.status != 200:
                raise DeliveryError("anonymous-asset-readback-status-not-200")
            while True:
                block = response.read(BLOCK_BYTES)
                if not block:
                    break
                size += len(block)
                if size > asset["bytes"]:
                    raise DeliveryError("anonymous-asset-readback-over-size")
                digest.update(block)
    except DeliveryError:
        raise
    except Exception:
        raise DeliveryError("anonymous-asset-readback-body-error-details-suppressed") from None
    if size != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
        raise DeliveryError("anonymous-asset-readback-size-or-sha256-mismatch")
    return digest.hexdigest()


def main() -> int:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise DeliveryError("actions-token-unavailable")
    assets, manifest_sha = _validate_manifest(token)
    total = sum(item["bytes"] for item in assets)
    results = []
    with tempfile.TemporaryDirectory(prefix="harbichess-release-allowlist-") as tmp:
        tmpdir = Path(tmp)
        for index, asset in enumerate(assets):
            existing = [a for a in _asset_list(token) if a.get("name") == asset["name"]]
            if len(existing) > 1:
                raise DeliveryError(f"existing-asset-name-duplicated-{index}")
            if existing:
                print(
                    json.dumps(
                        {
                            "event": "asset-found-pending-verification",
                            "name": asset["name"],
                            "asset_id": existing[0].get("id"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                sha = _public_asset_readback(asset, existing[0])
                result = {
                    "name": asset["name"],
                    "status": "existing-asset-readback-matched",
                    "bytes": asset["bytes"],
                    "sha256": sha,
                }
                results.append(result)
                print(
                    json.dumps({"event": "asset-readback-verified", **result}, sort_keys=True),
                    flush=True,
                )
                continue
            if shutil.disk_usage(tmpdir).free < asset["bytes"] + 256 * 1024**2:
                raise DeliveryError(f"temporary-disk-headroom-insufficient-{index}")
            path = tmpdir / f"asset-{index:03d}.bin"
            _download_source(asset, path)
            uploaded = _upload_file(asset, path, token)
            print(
                json.dumps(
                    {
                        "event": "asset-upload-accepted-pending-verification",
                        "name": asset["name"],
                        "asset_id": uploaded["id"],
                        "bytes": asset["bytes"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            matches = [a for a in _asset_list(token) if a.get("name") == asset["name"]]
            if len(matches) != 1 or matches[0].get("id") != uploaded.get("id"):
                raise DeliveryError(f"uploaded-asset-metadata-not-unique-{index}")
            readback_sha = _public_asset_readback(asset, matches[0])
            result = {
                "name": asset["name"],
                "status": "new-upload-readback-matched",
                "bytes": asset["bytes"],
                "sha256": readback_sha,
            }
            results.append(result)
            print(
                json.dumps({"event": "asset-readback-verified", **result}, sort_keys=True),
                flush=True,
            )
            path.unlink()
    print(
        json.dumps(
            {
                "status": "pass-all-manifest-assets-readback-verified",
                "repository": REPOSITORY,
                "release_id": RELEASE_ID,
                "release_tag": RELEASE_TAG,
                "manifest_sha256": manifest_sha,
                "asset_count": len(assets),
                "total_manifest_bytes": total,
                "assets": results,
                "claim_scope": (
                    "Only fixed manifest allowlist and matching public bytes; "
                    "no strength or checkpoint-completeness claim"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DeliveryError as exc:
        print(
            json.dumps({"status": "failed-preserved", "reason": exc.code}, sort_keys=True),
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception:
        print(
            json.dumps(
                {"status": "failed-preserved", "reason": "unexpected-error-details-suppressed"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
