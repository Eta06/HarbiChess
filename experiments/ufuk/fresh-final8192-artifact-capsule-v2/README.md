# FINAL8192 exact-byte artifact transport v2 — DRAFT, NOT DISPATCHED

Only four approved-scope originals: final actions-00008192.json.gz and immutable
actor-config.json for seeds20262805/06. These are final E0 milestones, not fits or
strength qualifications. The previous4096 unpublished v1 proposal remains untouched.
Source6fcc8b476d25495d1c9c413e55b2c7ba4794013e; exactE8 actor/anchor. Independent
finalreadiness SHA90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51
is sealed as provenance metadata, not raw Torch/NN output.

Actual raw originals885958B; deterministic regular-only capsule882330B SHA
3f6fef499cb166bbad466d4983954350df71b7337be4ceeff2d53ca2a2fb1271.
21chunks (43500rawB max,58000base64chars max);22serial workflow runs including
aggregate; ONE permanent content-addressed Release asset. Capsule/chunks were held
only in localRAM and no opaque data file/base64 was written in the proposal or Git.
Each original<=2MiB, total<=4MiB, capsule<=4MiB, temporaryZIP quota<=8MiB;
existing Release401698693 limits200assets/8GiB unchanged. No native state transformed.

Pinned actions: checkout11bd71901bbe5b1630ceea73d27597364c9af683,
upload-artifact ea165f8d65b6e75b540449e92b4886f43607fa02 (ROOT verifiedv4.6.2).
Workflow run-name contains mode/validated12-character manifestprefix/ordinal before
sealing. Fixed source/blob/manifest paths prevent arbitrary PATH/URL collection.
Root authorization is PENDING, allowlist EMPTY, manifest DRAFT, clocksNULL: code
cannot execute until exact prospectively reviewed approval/manifest are frozen.
Generic original1800s transportclock guard is reused from SHA-pinned frozen support;
it is not a producer/native schema or an extension of any old TRAIN clock.

## ROOT launch sequence

1. Review source/tests/actualfourfileSHA/manifest. Change only approval/status fields
   prospectively; regenerate exact manifestSHA, approval source seals and controlSHA.
   Do not change originalfiles/chunks or source/code after freeze. Authorizer admits
   one manifest SHA and literal manifest-8192-DRAFT.json only; rejects4096 schema,
   changed source/readiness/actionpins/anchors. Commit CODE+manifest/control/receipts
   only. No opaque payloads in Git. Upstream run headcommit source blobs must match
   exact receiver/support/workflow/manifestSHA; optional separately known commit pin
   avoids the impossible self-referential publication commit problem.
2. Register observed ROOTfirst and min(first+1800,2026-10-06T08UTC) ONCE for smoke,
   all21chunks and aggregate. No reset after retry/failure. Original train clocks
   and old transport assets remain immutable. Failclosed when time/cap exhausted.
3. Construct inputs with launcher_inputs.dispatch_inputs (no networking in helper).
   Dispatch chunk0 as actual transport smoke; record successful originalrunID,
   inspect one ZIP/regularchunk/metadata/size/SHA. It counts toward21; no duplicate.
4. Serialize ordinals1..20 under EXISTING a100-allowlisted-release-delivery mutex;
   wait each completed success, retain ordinal→runID receipt. GitHub retains only
   one pending run. Retention1day/compression0/nooverwrite. Duplicate temp names
   require reuse of original successfulrunIDs, never another artifact overwrite.
5. Aggregate21 unique IDs. Require exact upstream repository/event/path/sourceblob
   bindings, completed success, single named artifact/run binding, ZIPsize equals
   API size_in_bytes (failclosed; actualAPI behavior still must be tested), exact
   single regular chunk, eachSHA, fullcapsuleSHA and everyoriginalfileSHA.
6. Publish ONE matching-idempotent capsule to existingRelease; no delete/overwrite.
   Anonymous COMPLETEbodySHA and extracted fourSHAs required beforePASS. No auth
   on public download, no arbitrary redirects outside pinnedGitHub hosts. Temporary
   artifact success alone is not durable backup. Keep all failures/partial receipts.

28 localtests PASS0skip0.17s; RuffPASS. They cover source/schema/pin/readiness/path/
corruption/size/duplicate/upstream/emptyallowlist/streamclock/publicdownload bounds,
idempotentcaps and launcher payload envelope. Existing nativecodec/release support
copies are byte-identical ancestors. Actual Actions/upload/readback are UNEXECUTED;
public persistence, restored runtime and self-learning strength are NOT claimed.
