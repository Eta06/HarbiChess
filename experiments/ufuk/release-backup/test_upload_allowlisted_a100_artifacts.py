import importlib.util
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import ClassVar

import pytest

spec = importlib.util.spec_from_file_location(
    "delivery", Path(__file__).parents[3] / "scripts/upload_allowlisted_a100_artifacts.py"
)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


def _asset(asset_id, name=None):
    return {"id": asset_id, "name": name or f"asset-{asset_id}"}


def _link(page):
    return {
        "Link": (
            f"<https://api.github.com/repos/{delivery.REPOSITORY}/releases/"
            f'{delivery.RELEASE_ID}/assets?per_page=100&page={page}>; rel="next"'
        )
    }


def _page(path):
    return int(urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["page"][0])


def test_asset_list_follows_canonical_pages_and_collects_unique_assets(monkeypatch):
    first = [_asset(i) for i in range(1, 101)]
    second = [_asset(101)]
    requests = []

    def fake_request(path, token, *, include_headers=False):
        requests.append(path)
        if _page(path) == 1:
            return (first, _link(2))
        return (second, {})

    monkeypatch.setattr(delivery, "_api_request", fake_request)
    assert len(delivery._asset_list("test-token")) == 101
    assert len(requests) == 2
    assert "page=1" in requests[0] and "page=2" in requests[1]


def test_exactly_one_hundred_assets_without_link_probes_canonical_empty_page(monkeypatch):
    requests = []
    first = [_asset(i) for i in range(1, 101)]

    def request(path, token, *, include_headers=False):
        requests.append(path)
        assert include_headers and token == "test-token"
        return (first if _page(path) == 1 else [], {})

    monkeypatch.setattr(delivery, "_api_request", request)
    assert delivery._asset_list("test-token") == first
    base = f"/repos/{delivery.REPOSITORY}/releases/{delivery.RELEASE_ID}/assets?per_page=100&page="
    assert requests == [base + "1", base + "2"]


def test_asset_list_fails_closed_on_missing_page_or_duplicate_records(monkeypatch):
    first = [_asset(i) for i in range(1, 101)]

    def missing_page(path, token, *, include_headers=False):
        if _page(path) == 1:
            return (first, _link(2))
        raise delivery.DeliveryError("github-api-http-503")

    monkeypatch.setattr(delivery, "_api_request", missing_page)
    with pytest.raises(delivery.DeliveryError, match="github-api-http-503"):
        delivery._asset_list("test-token")

    def duplicate_page(path, token, *, include_headers=False):
        if _page(path) == 1:
            return (first, _link(2))
        return ([_asset(1)], {})

    monkeypatch.setattr(delivery, "_api_request", duplicate_page)
    with pytest.raises(delivery.DeliveryError, match="duplicated-across-pages"):
        delivery._asset_list("test-token")

    def noncanonical_next(path, token, *, include_headers=False):
        return (first, {"Link": '<https://evil.example/assets?page=2>; rel="next"'})

    monkeypatch.setattr(delivery, "_api_request", noncanonical_next)
    with pytest.raises(delivery.DeliveryError, match="next-link-not-canonical"):
        delivery._asset_list("test-token")


@pytest.mark.parametrize("include_link", [False, True])
def test_asset_list_stops_at_ten_pages_without_accepting_partial_listing(monkeypatch, include_link):
    requests = []

    def ten_full_pages(path, token, *, include_headers=False):
        page = _page(path)
        requests.append(path)
        assets = [_asset((page - 1) * 100 + i) for i in range(1, 101)]
        return (assets, _link(page + 1) if include_link else {})

    monkeypatch.setattr(delivery, "_api_request", ten_full_pages)
    with pytest.raises(delivery.DeliveryError, match="pagination-inconsistent-or-limit-reached"):
        delivery._asset_list("test-token")
    assert len(requests) == delivery.MAX_RELEASE_ASSET_PAGES


def test_api_path_is_canonical_and_never_follows_auth_redirect(monkeypatch):
    seen = {}

    class Response:
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return b"[]"

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr(delivery.NO_REDIRECT, "open", fake_open)
    path = f"/repos/{delivery.REPOSITORY}/releases/{delivery.RELEASE_ID}/assets?per_page=100&page=1"
    assert delivery._api_request(path, "synthetic-token", include_headers=True)[0] == []
    assert seen["url"] == "https://api.github.com" + path
    assert seen["authorization"] == "Bearer synthetic-token"
    with pytest.raises(delivery.DeliveryError, match="api-path-not-allowlisted"):
        delivery._api_request("/repos/other/repo/releases/123/assets", "synthetic-token")

    blocked = delivery.RejectRedirect().redirect_request(
        urllib.request.Request(
            "https://api.github.com/fixed", headers={"Authorization": "Bearer fake"}
        ),
        None,
        302,
        "Found",
        {},
        "https://api.github.com/redirected",
    )
    assert blocked is None


@pytest.mark.parametrize(
    "url",
    [
        "http://a.trycloudflare.com/asset.tar.gz",
        "https://a.trycloudflare.com:443/asset.tar.gz",
        "https://a.trycloudflare.com/other/asset.tar.gz",
        "https://a.trycloudflare.com/asset.tar.gz?x=1",
        "https://user@a.trycloudflare.com/asset.tar.gz",
        "https://a.trycloudflare.com.evil.test/asset.tar.gz",
    ],
)
def test_manifest_source_bounds_reject_noncanonical_routes(tmp_path, monkeypatch, url):
    digest = "a" * 64
    obj = {
        "schema": "harbichess-a100-release-transport-v1",
        "release_id": delivery.RELEASE_ID,
        "release_tag": delivery.RELEASE_TAG,
        "seeds": [20261205],
        "assets": [{"name": "asset.tar.gz", "bytes": 1, "sha256": digest, "url": url}],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(obj))
    monkeypatch.setattr(delivery, "MANIFEST", manifest)
    monkeypatch.setenv("GITHUB_REPOSITORY", delivery.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(
        delivery,
        "_api_request",
        lambda path, token: {
            "id": delivery.RELEASE_ID,
            "tag_name": delivery.RELEASE_TAG,
            "draft": False,
            "upload_url": f"https://{delivery.UPLOAD_HOST}/repos/{delivery.REPOSITORY}/releases/{delivery.RELEASE_ID}/assets{{?name,label}}",
            "assets_url": (
                f"{delivery.API_ROOT}/repos/{delivery.REPOSITORY}/releases/"
                f"{delivery.RELEASE_ID}/assets"
            ),
        },
    )
    with pytest.raises(delivery.DeliveryError, match="source-url-not-fixed"):
        delivery._validate_manifest("test-token")


def test_packager_style_manifest_with_seed_metadata_is_accepted(tmp_path, monkeypatch):
    assets = []
    for index in range(101):
        digest = f"{index:064x}"
        name = f"sha256-{digest}.tar.gz"
        assets.append(
            {
                "name": name,
                "bytes": 123,
                "sha256": digest,
                "url": f"https://a123.trycloudflare.com/{name}",
            }
        )
    obj = {
        "schema": "harbichess-a100-release-transport-v1",
        "release_id": delivery.RELEASE_ID,
        "release_tag": delivery.RELEASE_TAG,
        "seeds": [20261205, 20261206],
        "assets": assets,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(obj))
    monkeypatch.setattr(delivery, "MANIFEST", manifest)
    monkeypatch.setenv("GITHUB_REPOSITORY", delivery.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(
        delivery,
        "_api_request",
        lambda path, token: {
            "id": delivery.RELEASE_ID,
            "tag_name": delivery.RELEASE_TAG,
            "draft": False,
            "upload_url": f"https://{delivery.UPLOAD_HOST}/repos/{delivery.REPOSITORY}/releases/{delivery.RELEASE_ID}/assets{{?name,label}}",
            "assets_url": (
                f"{delivery.API_ROOT}/repos/{delivery.REPOSITORY}/releases/"
                f"{delivery.RELEASE_ID}/assets"
            ),
        },
    )
    assets, digest_out = delivery._validate_manifest("test-token")
    assert len(assets) == 101 and digest_out
    assert delivery.MAX_ASSETS == 200


def test_public_asset_redirect_strips_auth_and_rejects_unapproved_host():
    request = urllib.request.Request(
        "https://api.github.com/repos/Eta06/HarbiChess/releases/assets/1",
        headers={"Authorization": "Bearer never-forward"},
    )
    redirect = delivery.PublicGitHubRedirect()
    clean = redirect.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://release-assets.githubusercontent.com/file?sig=public",
    )
    assert clean is not None and clean.get_header("Authorization") is None
    with pytest.raises(urllib.error.HTTPError):
        redirect.redirect_request(request, None, 302, "Found", {}, "https://evil.example/asset")
    with pytest.raises(urllib.error.HTTPError):
        redirect.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com:443/asset",
        )


def test_source_download_sends_no_auth_and_rejects_redirect(tmp_path, monkeypatch):
    seen = {}

    def fake_open(request, timeout):
        seen["authorization"] = request.get_header("Authorization")
        raise urllib.error.HTTPError(request.full_url, 302, "redirect", {}, None)

    monkeypatch.setattr(delivery.NO_REDIRECT, "open", fake_open)
    digest = "c" * 64
    asset = {
        "url": f"https://a123.trycloudflare.com/sha256-{digest}.tar.gz",
        "bytes": 1,
        "sha256": digest,
    }
    with pytest.raises(delivery.DeliveryError, match="redirects-disallowed"):
        delivery._download_source(asset, tmp_path / "never-created.bin")
    assert seen["authorization"] is None


def test_actual_numeric_repository_link_never_becomes_a_credentialed_request(monkeypatch):
    first = [_asset(i) for i in range(1, 101)]
    requests = []
    actual_link = (
        "<https://api.github.com/repositories/1345115839/releases/401698693/"
        'assets?per_page=100&page=2>; rel="next", '
        "<https://api.github.com/repositories/1345115839/releases/401698693/"
        'assets?per_page=100&page=2>; rel="last"'
    )

    def request(path, token, *, include_headers=False):
        requests.append(path)
        assert token == "test-token" and include_headers
        return (first, {"Link": actual_link}) if _page(path) == 1 else ([_asset(101)], {})

    monkeypatch.setattr(delivery, "_api_request", request)
    assert len(delivery._asset_list("test-token")) == 101
    base = "/repos/Eta06/HarbiChess/releases/401698693/assets?per_page=100&page="
    assert requests == [base + "1", base + "2"]


@pytest.mark.parametrize(
    "target",
    [
        "https://api.github.com/repositories/1345115840/releases/401698693/assets?per_page=100&page=2",
        "https://api.github.com/repositories/1345115839/releases/401698694/assets?per_page=100&page=2",
        "https://evil.example/repositories/1345115839/releases/401698693/assets?per_page=100&page=2",
        "https://user@api.github.com/repositories/1345115839/releases/401698693/assets?per_page=100&page=2",
        "https://api.github.com/repositories/1345115839/releases/401698693/assets?per_page=100&page=2&page=3",
        "https://api.github.com/repositories/1345115839/releases/401698693/assets?per_page=100&page=2&token=x",
        "https://api.github.com/repositories/1345115839/releases/401698693/assets?per_page=100&page=3",
    ],
)
def test_numeric_link_foreign_identity_or_noncanonical_query_fails_before_next_request(
    monkeypatch, target
):
    requests = []

    def request(path, token, *, include_headers=False):
        requests.append(path)
        return ([_asset(i) for i in range(1, 101)], {"Link": f'<{target}>; rel="next"'})

    monkeypatch.setattr(delivery, "_api_request", request)
    with pytest.raises(delivery.DeliveryError):
        delivery._asset_list("test-token")
    assert requests == ["/repos/Eta06/HarbiChess/releases/401698693/assets?per_page=100&page=1"]
