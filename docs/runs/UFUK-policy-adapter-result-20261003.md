# UFUK: value-safe policy representation gate failed

The additional policy residual branch did not materially beat the head-only
control. Both learned the fixed own-search targets, but heldout policy CE
2.38602755(control best8000) versus2.38586425(adapter best4000) differs by only
0.00016330, below the preregistered0.03. Representation gate **failed**; no arena,
promotion or extended lucky-seed run. This is not a played-strength regression
claim: no games were played with these candidates.

[Preregistration](UFUK-policy-adapter-preregistration-20261003.md), clean pinned
source59dde8df863d4306bc7e210454ad0e94829b24d8. Original native model
SHA18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae.
Weights-only identity transfer, optimizer reset for both arms; within-task native
full optimizer/RNG/sampler/input-checksum restore in separate processes at250.
New optional portable specification policy_adapter={schema:1,blocks:2}; weight
container1, encoder/action/WDL and original replay schemas unchanged. Old plain
models load unchanged; unknown adapter schemas fail loudly.

| Measurement | Head-only control | Policy-only adapter |
| --- | --- | --- |
| Total/trainable parameters |72,497/16,914|81,777/26,194|
| Inherited frozen parameters |55,583|55,583|
| Actual updates |8000|max5500,patience1500 afterbest4000|
| Sampled own-search rows |512,000|352,000|
| Heldout initial policy CE |2.46256582|2.46256582|
| Best heldout policy CE |2.38602755|2.38586425|
| Gain from initial |0.07653827|0.07670157|
| Native20k policy CE |2.74479786|2.76190857|
| Native policy CE change |+0.11269851|+0.12980923|
| Native value CE / Q MAE |0.65434915/0.33688294|exactly same|
| Native top16 teacher coverage |0.89580|0.89385|
| Training outer wall |283.940385s|220.462899s|

Both >=0.05 self-target improvement and <=0.15 native-policy retention passed;
the additional-representation comparison failed. Different early-stop lengths
mean total wall is not equal-workload inference/training speed evidence. All23
common checkpoints0..5500 have identical complete sample-trace hashes, Python
sampler states and Torch RNG states. Every250 updates verifies frozen parameter
hashes and bitwise value predictions. Shared original trunk remains frozen,
since it feeds base value; only local policy copies enter the adapter.

72train/24validation games,36/12 existing actor families,11,547/3338 records,
5979/2178 known terminal targets. Entire histories disjoint, all source shards
checksummed. Existing native-development families were previously seen, so this
is not independent generalization. Original unknown outcomes remain mask0;
heldout self value CE1.54338922 is constant, not learned. Native finite-teacher
WDL and weak-policy actual terminal outcomes describe different continuation
policies; their losses are not interchangeable.

Fixed game-balanced sampling seed20261012/batch64/AdamW5e-5/wd1e-4/clip5/CPU1
FP32. Setup cache49.660975s is separately recorded and conservatively charged to
each1800s cap. Control phase parent CPU23.714874+255.232315s; adapter
24.632062+190.544839s; peakRSS1,317,304/1,339,732KiB. Existing
4CPU/16GiB only, no new paid compute. Native qualification observer20k rows is
never a training target. Cache checksum b12f6a5a85aa914414084ec1532ae6221d0e37ec0a7770bfc80ef46eb870bd89;
preserve the exact training cache for original-source full resume.

Best control weights SHAef2ee1d26bb9b0e80a9e849009c7ed326b43afe9a9881f0160ad2cb14a6dc96a;
best adapter SHA0b92a4b0c8f71ea71b354b19c1acdd52c96656a18a1db3eabf64ea4a48a2e52a.
All intermediate/full checkpoints, replays and initial model retained.
501tests passed/zero skips67.64s; real MLXCPU full/masked forward,loss/gradients,
actual updates preserving value, portable MLX round-trip and Torch native resume.
Two test fixture failures were corrected and initial logs retained; no skips.
Apple Metal/CUDA hardware remains untested.

[Exact text evidence](UFUK-policy-adapter-evidence-20261003.json) contains commands,
curves, full-checkpoint metadata, cache/source manifests, test/preflight logs and
registered gate. Binary Release backup remains incomplete with400 responses.
The target signal is positive but extra depth/representation was insufficient;
next isolate conventional search breadth versus depth at fixed16sim, rather than
claiming that more layers or arbitrary longer training must yield chess strength.
