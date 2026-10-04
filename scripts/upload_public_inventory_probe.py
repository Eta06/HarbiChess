"""Upload one fixed public manifest to one existing release, then verify anonymously."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "Eta06/HarbiChess"
RELEASE_ID = 401698693
RELEASE_TAG = "port-linux-preflight-20261002"
MANIFEST = Path("docs/runs/UFUK-A100-parent-inventory-transport-proof-20261004.json")
ASSET_NAME = "UFUK-A100-parent-inventory-transport-proof-20261004.json"
EXPECTED_BYTES = 543
EXPECTED_SHA256 = "a7b139423f04a18f974bcbfdbf5248800d2624e8680aa85350dfbff29d0f320c"
API_ROOT = "https://api.github.com"
UPLOAD_HOST = "uploads.github.com"
USER_AGENT = "HarbiChess-public-inventory-release-probe/1.0"
API_VERSION = "2022-11-28"
SAFE_DOWNLOAD_HOSTS = {
    "api.github.com",
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


def request(
    url: str,
    *,
    token: str | None = None,
    accept: str = "application/vnd.github+json",
    data: bytes | None = None,
    method: str | None = None,
    opener=None,
    content_type: str = "application/json",
):
    headers = {"User-Agent": USER_AGENT, "Accept": accept, "X-GitHub-Api-Version": API_VERSION}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(data))
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    if opener is not None:
        return opener.open(req, timeout=30)
    return urllib.request.urlopen(req, timeout=30)


def api_json(path: str, token: str) -> dict | list:
    with request(API_ROOT + path, token=token) as response:
        return json.load(response)


class PublicDownloadRedirect(urllib.request.HTTPRedirectHandler):
    """Allow only HTTPS redirects among GitHub asset hosts; never forward headers."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (
            old.scheme != "https"
            or new.scheme != "https"
            or old.hostname not in SAFE_DOWNLOAD_HOSTS
            or new.hostname not in SAFE_DOWNLOAD_HOSTS
            or new.username is not None
            or new.password is not None
        ):
            raise urllib.error.HTTPError(
                newurl, code, "Rejected non-GitHub asset redirect", headers, fp
            )
        # Intentionally create a fresh request without Authorization or API headers.
        return urllib.request.Request(newurl, headers={"User-Agent": USER_AGENT}, method="GET")


def main() -> int:
    if (
        os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
    ):
        raise RuntimeError(
            "This fixed probe only runs by manual dispatch on the canonical main branch"
        )
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("The Actions job token is unavailable")

    payload = MANIFEST.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise RuntimeError("Tracked manifest bytes do not match the fixed public input")

    release = api_json(f"/repos/{REPOSITORY}/releases/{RELEASE_ID}", token)
    expected_upload_url = (
        f"https://{UPLOAD_HOST}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets{{?name,label}}"
    )
    expected_assets_url = f"{API_ROOT}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets"
    if (
        release.get("id") != RELEASE_ID
        or release.get("tag_name") != RELEASE_TAG
        or release.get("draft") is not False
        or release.get("upload_url") != expected_upload_url
        or release.get("assets_url") != expected_assets_url
    ):
        raise RuntimeError(
            "Public release identity or upload endpoints differ from the fixed target"
        )

    assets = api_json(f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets", token)
    if any(asset.get("name") == ASSET_NAME for asset in assets):
        raise RuntimeError("Refusing to overwrite an existing release asset")

    query = urllib.parse.urlencode({"name": ASSET_NAME, "label": ASSET_NAME})
    upload_url = f"https://{UPLOAD_HOST}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets?{query}"
    with request(
        upload_url,
        token=token,
        data=payload,
        method="POST",
        content_type="application/octet-stream",
    ) as response:
        uploaded = json.load(response)
    if uploaded.get("name") != ASSET_NAME or uploaded.get("size") != EXPECTED_BYTES:
        raise RuntimeError("Upload response did not match the fixed asset identity")
    asset_id = uploaded.get("id")
    if type(asset_id) is not int or asset_id <= 0:
        raise RuntimeError("Upload response did not include a valid asset ID")

    checked_assets = api_json(f"/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets", token)
    matches = [asset for asset in checked_assets if asset.get("name") == ASSET_NAME]
    if (
        len(matches) != 1
        or matches[0].get("id") != asset_id
        or matches[0].get("size") != EXPECTED_BYTES
    ):
        raise RuntimeError("Release asset metadata readback was not unique and exact")
    asset_url = matches[0].get("url")
    if asset_url != f"{API_ROOT}/repos/{REPOSITORY}/releases/{RELEASE_ID}/assets/{asset_id}":
        raise RuntimeError("Readback asset API URL was not the expected GitHub release endpoint")

    # Separate unauthenticated GET; redirects are restricted to HTTPS GitHub hosts and
    # the handler discards all request headers so no token can reach the asset CDN.
    public_opener = urllib.request.build_opener(PublicDownloadRedirect())
    with request(asset_url, accept="application/octet-stream", opener=public_opener) as response:
        readback = response.read(EXPECTED_BYTES + 1)
    readback_digest = hashlib.sha256(readback).hexdigest()
    if len(readback) != EXPECTED_BYTES or readback_digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Unauthenticated release-asset download failed exact size/SHA verification"
        )

    print(
        json.dumps(
            {
                "status": "pass-uploaded-and-anonymous-readback-verified",
                "repository": REPOSITORY,
                "release_id": RELEASE_ID,
                "release_tag": RELEASE_TAG,
                "asset_name": ASSET_NAME,
                "asset_id": asset_id,
                "bytes": len(readback),
                "sha256": readback_digest,
                "auth_scope": (
                    "Actions GITHUB_TOKEN in this fixed contents:write job only; "
                    "not forwarded to download or A100"
                ),
                "claim_scope": "543-byte public manifest only; not a checkpoint or replay backup",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
