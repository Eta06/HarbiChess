# Fresh own-search consistency trainer — scratch, unregistered

Separate native `fresh-qsearch-additive-own-sc-native-v1`, no oldMC resume alias.
This is the proposed fixed .5own-terminal CE+.5MSE(pW-pL,clamped recordedselected
rootQ)+1frozenE8 KL+unchangedSHRINK penalty. SearchQ is pre-action rootmover POV,
selected-best move distinct from5%exploratory playedaction. CompleteKNOWN games
ONLY; caps/tails UNKNOWN excluded from BOTH losses. No external teacher, new
search, actor generation, policy gradient or offpolicy-unbiased-return claim.
Standard ownsearch selfdistillation/ExIt; no novelty or strength claim.

Derived from CORRECTED actualMAIN MC trainer SHA49445cbd5059770b3ccd433be9f21a4773786885a85d2d586005e10e2359a2c6.
`frozen-mc-parent-corrected.py.txt` preserves that source. Prior scratch parent
1df7a208 is preserved separately and was not falsely claimed compatible with the
actualMAIN producer. Its converter expected different receipt names; rootfixed MC
consumer accepts actual `fresh-qsearch-to-shrink840-handoff-v2`, `anchor_model_sha256`
and exactanchorhelper/target, with rowcount verified from arrays. Producer unchanged.
Ten common initialization/data/filter/split/checkpoint functions remain exactAST
copies of correctedMC. Core source6fcc8b476d25495d1c9c413e55b2c7ba4794013e unchanged.

SC uses identical prepare_fresh data, protectedFEN fulltrajectory exclusion,
realized-trajectory dedup/hashsplit, complete-game uniform sampler then rowuniform,
256slots/update and min(1024,floor(4*TRAINrows/256)). Zeroadditive initialization,
AdamW2e-5/decay0/foreachFalse, clipping5 and original coordinate/metadata masks are
unchanged. Frozen E8/storage bits including unused21 material coordinates are
verified without any nonzero assumption. Only sparse feature+bias+metadata train.

Composition hazard prevented: original critic already returns log(E8)+delta.
SC extracts the exact delta directly and helperadds log(E8) ONCE. A storage-bit
regression compares SC-composed logits to originalcritic. Anchors/searchtargets
are detached. No subtraction/readdition roundoff route is used.

Native saves headstate, Adam, allCPU Torch/Python/NumPy/sampler RNG, updatecursor,
trainablemask, commonMC datasetSHA, extendedtargetdatasetSHA, raw/normalizedtarget
SHA and selected/played fullhistory row-ledgerSHA, allinput/helper/protocol SHAs.
Frozen backbone reconstructed only from exactSHA originalE8 plus pinnedcore/rebase;
THIS IS OFFLINE RESUME ONLY, no activeactor/searchstate claim. It is TWO regular
native files (`training.pt`,`checkpoint.json`), not sixpayloads or fullonline state.
Save/resume rejects SC/MC schema aliases and altered data/targets/source/contract.
SC explicitlychecks importedcore moduleorigin inside SHA-clean sourcecheckout,
plus locallySHA-pinned features/sc helper. Native manifestmask nowchecked too.

`qualify.py` is the REAL-data prospective8whole/4pause/freshresume8 controller with
six independentfresh strictloads at0/4/8 whole+split. It has NOT been executed on
realfresh8192/activejournals. Rootmust freeze newprotocol/clocks/endpoints before
that qualification or anyrealfit. Exactsame rowdata and split, no validation-based
selection. Actual firsteligible2048 proof and later8192 endpoint are rootowned.

LOCAL6testsPASS28.22s/RuffPASS, CPU1. Legalconverter tests use immutableMAIN v2
producer, exact840 features, reallegalhistories/protectedfulltrajectory/dedup/split.
Freshprocess native test runs actualSC CLI/update/checkpoint/strictload paths with
an EXPLICIT SYNTHETIC common-data seam; realE8 backbone only, no NNactor/search or
realselfplay dataset fit. Whole8/pause4/resume8 restored allhead/Adam/global+sampler
RNG/data/contract payloads storage-exact at0/4/8; all6fresh strictloads passed. Its
synthetic source/protocol cannot match an actualdata contract and is not eligible.
Step0/4 rawTorchZIP bytes agreed; step8 differed in serialization representation
while complete restored trees were exact. Initial overstrong rawbyte failure and
intermediateproofs are preserved. IndividualartifactSHAs remain mandatory; no
rawZIP identity claim is made. Tempnative unitfiles lived inRAM and were cleaned
by their ownedtest context. Receipt/provenance retained, no oldfiles touched.

No main/activehelper/Git/network/fit/selfplay/matches/GPU mutation launched.
