# FINAL8192 public Release recovery v3 — UNEXECUTED

Explicit transport recovery only. Failedv2 chunk0 Actions run37360596243 and
artifact11365929280 are retained. Azure CONNECT403 is preserved, not bypassed.
This code DOES NOT retrieve Actions artifacts or Azure blobs, use SSH/Docker/
remote fetching, alter proxies, or request another machine. It uses LOCAL original
bytes and the existing authorized GitHub Actions→PUBLIC Release delivery route.
Old v1/v2 evidence, codecs, assets and actor/training contracts remain untouched.

Four originals are identical to v2: both final8192journals + immutable actorconfigs
20262805/06, raw885958B. Recomputed exact deterministic capsule882330B SHA
3f6fef499cb166bbad466d4983954350df71b7337be4ceeff2d53ca2a2fb1271;
all21originalrawchunk SHA/sizes identical. Opaque capsule/chunks held only in RAM,
never saved in proposal/Git. Fixedpaths, finalschema, source6fcc/E8/readiness901624,
receiver/frozen support/workflow SHAs unchanged in rigor. Only transport/version
changes. Checkout action pinned11bd71901bbe5b1630ceea73d27597364c9af683; no
upload-artifact action/permission or artifact download API is needed.

**Original sharedclock MUST stay first1791226974.7925532→end1791228774.7925532.**
Code explicitly rejects any replacement clock. Every smoke/chunk/aggregate retry
uses that clock and operator08UTC ceiling. If insufficient time/expired, retain
INCOMPLETE; never reset. ROOT approvals remainPENDING and allowlistEMPTY.

## ROOT publication/setup recipe

1. Review source/tests/newv3manifest and unchanged fourfile/capsule/chunkSHA.
   Set manifeststatus ROOT-approved-frozen-transport, approvalstatus
   ROOT-reviewed-prospectively-approved. Recompute exact approvedmanifestSHA,
   ROOT-approval source seals/control manifestSHA; leave the above clock literal.
   Generate fixedallowlist with authorize_manifest.py. Freeze all code/helper/
   workflow SHA. Commit smallCODE/manifest/control/failureprovenance only; no opaque
   journals/chunks/RNG/payloads in Git. v3approval is separate from preservedv2.
2. Dispatch chunk0 through launcher_inputs.py using newmanifestSHA/prefix + rawchunk
   base64payload (<=58000chars; totalJSON<=65535). Wait actual completed success.
   Receiver decodes/validates rawchunkSHA and publishes immutable name
   fresh8192v3-<capsuleSHA>-chunk-000.bin via frozen release.publish_verified.
   Full anonymous public-body SHA is required even for a matching retry.
3. Dispatch ordinals1..20 SERIAL under existing a100-allowlisted-release-delivery
   mutex. Never queue many pending runs. All still use originalfirst+1800; no timer
   reset or alternatearchive. Partial chunks persist as actual partialbackup.
4. Dispatch aggregate (no payload/runIDs). Canonical existing Release inventory,
   each exactnamed chunk metadata/size/SHA,21anonymous PUBLIC GitHub downloads,
   totalRAM<=4MiB and old8MiBbound, fullcapsuleSHA plus fouroriginalregularfileSHAs
   must agree. Publish ONE sha256-<capsuleSHA>.tar.gz, matching retry/nooverwrite.
   Anonymous complete capsule SHA + extracted fourSHAs required beforebackupPASS.
   Checked-out workflow SHA is verified before network use. Any deadline failure
   remains failure even if some assets exist.

21chunks+1capsule=22new permanentassets: reported current166→188<=200, extra
1764660rawbytes +metadata safely below existing8GiB. Actual inventory/caps rechecked
on EACH upload; oldassets never renamed/deleted/overwritten. Unlike v2, rawsmall
chunks are PUBLIC Release assets (not expiring Azure artifacts). This is an explicit
approved asset-budget tradeoff, not a silent widening or clock extension.

24 localtests PASS0skip0.16s/RuffPASS: exact fixedcapsule, version/source/final/caps,
SHA/corruption/path/size/duplicate/missing/foreignURL/originalclockexpiry rejection,
matching-retry/nooverwrite, public overlong/shortbody, fixedallowlist/authorizer,
JSONdispatchbounds. No actual network/dispatch/Git/job by this agent. No public
persistence/restored runtime/strength claim until ROOTactualanonymous proof.
