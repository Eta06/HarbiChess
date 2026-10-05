# Exact native backup extension v2 — sealed transport

This version seals only final step1024 SHRUNK seeds20262605/06 and
RESIDUAL seeds20262705/06. All three original files are reconstructed byte for
byte: model.safetensors, training.pt (including Adam/RNG/pickles/ZIP metadata),
checkpoint.json. No Torch import, unpickling, inference, optimizer step, network,
upload, workflow dispatch or Git change was executed.

The generic lossless codec is identical to v1, SHA878868940b5bae15c182dce55215320b5f0ba3692b9c5eac26988204832a31d6.
The new binding adapter loads a private instance and accepts only the two new
schemas/four seeds/final1024. SHRUNK source4cae08522ac940746cf9127ca8ce4a6b63a47408;
RESIDUAL source6fcc8b476d25495d1c9c413e55b2c7ba4794013e. Original contracts are
bound completely; actual protocol/helper/features SHAs were checked separately.
No original native source is relabeled. Old v1 module/receiver/workflow/control,
assets and manifests are untouched.

Measured packets:78690,78758,84222,84265 bytes;325935 bytes total, eight chunks
(two per native). Every ASCII chunk <=58000 characters, every dispatch envelope
<=65535 bytes. All12 raw files matched full SHA after local reconstruction.
CPU1 roundtrip took0.4297seconds.35 meaningful tests passed1.34seconds; RuffPASS.
Tests include original raw-file CRC/tensor/parent/path/bounds corruption,
private-module isolation, fixed-family/source rejection, idempotent upload
mocks, no overwrite/deletion, anonymous full-byte corruption, and unchanged caps.

Suggested repository destination: experiments/ufuk/native-delta-extension-v2;
workflow .github/workflows/cpu-native-delta-release-v2.yml. Publish only code,
tests, manifests and receipts. Never put packet.zlib/chunk files/raw native RNG
into Git. The RAM proposal is below16MiB and work/ links to that RAM directory.

Root froze the new prospective1800-second transport clock in
transport-control-PENDING.json, statusROOT-approved-frozen-transport, before
publishing/dispatch. Existing Release401698693, same public E8 parent archive,
200asset/8GiB aggregate caps, no oldasset overwrite or deletion. At parentreported
154 existingassets, eight chunks+four capsules would reach166; receiver checks
actual live inventory. Serially dispatch/wait each completion: shared GitHub
concurrency permits onlyone pending run and a later pending run cancels its
predecessor. After chunks, aggregate each packet with empty chunk inputs.

Only actual final anonymous capsule SHA and extracted three raw-file SHAs prove
public persistence. Local roundtrip is not public or strict runtime resume proof.
A separate read-only restore audit should load downloaded states under each
actual producer/helper and compare dataset/split/frozen/model/Adam/RNG/counters,
without replacing old expired training deadlines. Fresh selfplay journals require
a future separate explicit dataset extension; they are outside this allowlist.

Root integration preserved an initial missing-external-chunk fixture test failure. No opaque native payload was added to Git. With the sealed external fixture directory explicitly supplied via HARBICHESS_NATIVE_V2_FIXTURES=/workspace/work/harbichess/native-delta-extension-v2-proposal, the 35 backup tests plus16 readonly restore tests passed (51 total,3.57s) without skips; Ruff passed. Reproduction requires the four exact sealed payload fixtures and E8 parent file, or their byte-identical public downloads, rather than treating missing artifacts as a passing test.
