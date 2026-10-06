"""Fixed new-release transport, bounded GET retries, single-attempt POSTs."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import raw_capsule_v9 as codec

REPO = "Eta06/HarbiChess"
API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"
END = 1791273600  # 2026-10-06 08:00 UTC / 11:00 Istanbul
ASSET_CAP = 200
BYTE_CAP = 8 * 1024**3
OLD_RELEASE = 401698693
OLD_TAG = "port-linux-preflight-20261002"
HOSTS = {
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


class Failure(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise Failure(code)


def check_clock(control, now=None):
    now = time.time() if now is None else now
    require(
        control.get("schema") == "classical-raw-blob-release-clock-v9"
        and control.get("status") == "ROOT-approved-frozen-transport"
        and control.get("operator_end_epoch") == END,
        "unapproved-new-scope-control",
    )
    start, end = control["started_epoch"], control["deadline_epoch"]
    require(
        type(start) in (int, float)
        and type(end) in (int, float)
        and start <= now < end <= min(start + 5400, END),
        "new-scope-clock-expired-or-invalid",
    )
    return min(15, end - now)


def safe_name(name):
    require(
        isinstance(name, str)
        and name
        and len(name) <= 180
        and all(
            c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._" for c in name
        ),
        "asset-name",
    )
    return name


def public_url(tag, name):
    safe_name(tag)
    safe_name(name)
    return f"https://github.com/{REPO}/releases/download/{tag}/{name}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urllib.parse.urlsplit(newurl)
        require(
            p.scheme == "https"
            and p.netloc == p.hostname
            and p.hostname in HOSTS
            and not p.fragment,
            "public-redirect-host",
        )
        return urllib.request.Request(
            newurl, headers={"User-Agent": "HarbiChess-state-v6"}, method="GET"
        )


class Transport:
    def __init__(self, token, control, release_binding):
        check_clock(control)
        require(
            set(release_binding) == {"id", "tag", "target_commitish", "initial_assets"},
            "fixed-new-release-binding-fields",
        )
        require(
            type(release_binding["id"]) is int
            and release_binding["id"] > 0
            and release_binding["id"] != OLD_RELEASE
            and release_binding["initial_assets"] == 1
            and release_binding["id"] == 404068972
            and release_binding["tag"] == "ufuk-cpu-ownplay-state-20261005"
            and release_binding["target_commitish"] == "31a18e981e27e696fcca69c7e62210e7f0e89830",
            "new-release-not-approved-or-old-target",
        )
        safe_name(release_binding["tag"])
        commit = release_binding["target_commitish"]
        require(
            isinstance(commit, str)
            and len(commit) == 40
            and all(c in "0123456789abcdef" for c in commit),
            "release-commit-binding",
        )
        self.token, self.control, self.binding = token, control, release_binding
        self.api_opener = urllib.request.build_opener(NoRedirect())
        self.public_opener = urllib.request.build_opener(PublicRedirect())

    def request(self, url, data=None, public=False):
        # Fixed-path callers construct URLs. Public requests never contain Authorization.
        parsed = urllib.parse.urlsplit(url)
        require(
            parsed.scheme == "https"
            and parsed.netloc == parsed.hostname
            and parsed.hostname
            in ({"github.com"} if public else {"api.github.com", "uploads.github.com"}),
            "request-host",
        )
        for attempt in range(1 if data is not None else 3):
            timeout = check_clock(self.control)
            headers = {"User-Agent": "HarbiChess-state-v6"}
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
                    {
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(data)),
                    }
                )
            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method="POST" if data is not None else "GET",
            )
            try:
                return (self.public_opener if public else self.api_opener).open(
                    request, timeout=timeout
                )
            except urllib.error.HTTPError as error:
                if data is not None:
                    raise Failure("post-unconfirmed-no-automatic-retry") from None
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise Failure(f"get-http-{error.code}-details-suppressed") from None
            except Failure:
                raise
            except Exception:
                if data is not None:
                    raise Failure("post-ambiguous-no-automatic-retry") from None
                if attempt == 2:
                    raise Failure("get-three-attempts-failed-details-suppressed") from None
        raise Failure("request-unreachable")

    def api_json(self, path, data=None):
        ident = self.binding["id"]
        base = f"/repos/{REPO}/releases/{ident}"
        require(path == base or path.startswith(base + "/assets?"), "fixed-api-path")
        with self.request((UPLOAD if data is not None else API) + path, data=data) as response:
            raw = response.read(256 * 1024 + 1)
        require(len(raw) <= 256 * 1024, "api256KiB")
        return json.loads(raw)

    def inventory(self):
        ident = self.binding["id"]
        row = self.api_json(f"/repos/{REPO}/releases/{ident}")
        require(
            row.get("id") == ident
            and row.get("tag_name") == self.binding["tag"]
            and row.get("target_commitish") == self.binding["target_commitish"]
            and row.get("draft") is False
            and row.get("upload_url")
            == f"{UPLOAD}/repos/{REPO}/releases/{ident}/assets{{?name,label}}",
            "new-public-release-identity",
        )
        rows, ids, names = [], set(), set()
        for page in range(1, 22):  # Exact 200 requires page21 empty; never follow Link URLs.
            batch = self.api_json(f"/repos/{REPO}/releases/{ident}/assets?per_page=10&page={page}")
            require(isinstance(batch, list) and len(batch) <= 10, "canonical-page10")
            for asset in batch:
                require(
                    type(asset.get("id")) is int
                    and asset["id"] > 0
                    and asset["id"] not in ids
                    and asset.get("name") not in names
                    and type(asset.get("size")) is int
                    and asset["size"] >= 0,
                    "inventory-duplicate-or-invalid",
                )
                ids.add(asset["id"])
                names.add(asset["name"])
            rows.extend(batch)
            require(
                len(rows) <= ASSET_CAP and sum(a["size"] for a in rows) <= BYTE_CAP,
                "caps200-8GiB",
            )
            if len(batch) < 10:
                protected = [
                    a
                    for a in rows
                    if a["name"]
                    == (
                        "freshstatev6-sha256-"
                        "66ef6b4f05a9ff3d404debb424d477860b2a7439c7ce51fe91c4c6d3399409ff.bundle.zlib"
                    )
                ]
                require(
                    len(protected) == 1
                    and protected[0]["size"] == 1845748
                    and protected[0].get("state") == "uploaded",
                    "existing-v6-asset-must-remain",
                )
                return rows
        raise Failure("inventory-incomplete")

    def asset(self, row, name, raw):
        ident = row.get("id")
        digest = codec.digest(raw)
        require(
            type(ident) is int
            and ident > 0
            and row.get("name") == name
            and row.get("size") == len(raw)
            and row.get("state") == "uploaded"
            and row.get("url") == f"{API}/repos/{REPO}/releases/assets/{ident}"
            and row.get("browser_download_url") == public_url(self.binding["tag"], name)
            and row.get("digest") in (None, "sha256:" + digest),
            "asset-identity-bytes-sha",
        )

    def upload(self, name, raw):
        query = urllib.parse.urlencode({"name": safe_name(name), "label": name})
        return self.api_json(
            f"/repos/{REPO}/releases/{self.binding['id']}/assets?{query}", data=raw
        )

    def download(self, name, size):
        require(type(size) is int and 0 < size <= 64 * 1024**2, "public-capsule-approved64MiB")
        with self.request(public_url(self.binding["tag"], name), public=True) as response:
            raw = response.read(size + 1)
        require(len(raw) == size, "public-full-body-size")
        return raw


def publish_verified(transport, name, raw):
    inventory = transport.inventory()
    matches = [a for a in inventory if a["name"] == name]
    require(len(matches) <= 1, "duplicate-name")
    if matches:
        asset, disposition = matches[0], "matching-retry-no-write"
    else:
        require(
            len(inventory) + 1 <= ASSET_CAP
            and sum(a["size"] for a in inventory) + len(raw) <= BYTE_CAP,
            "caps-before-upload",
        )
        asset, disposition = transport.upload(name, raw), "new-upload"
        after = [a for a in transport.inventory() if a["name"] == name]
        require(len(after) == 1 and after[0]["id"] == asset["id"], "unique-new-upload")
    transport.asset(asset, name, raw)
    require(
        codec.digest(transport.download(name, len(raw))) == codec.digest(raw),
        "anonymous-full-byte-sha",
    )
    check_clock(transport.control)
    return {
        "name": name,
        "asset_id": asset["id"],
        "bytes": len(raw),
        "sha256": codec.digest(raw),
        "disposition": disposition,
        "anonymous_full_body_sha_verified": True,
    }
