# UFUK main40: independent self-learning strength FAILED

Both prospectively frozen 40-epoch candidates failed the independent strength gates. The strongest validated checkpoint remains teacher-origin e8; these self-play-trained candidates are not promoted.

Producer: `2312652dc52a894e9726f48321117cf114270355`. Seeds20261205/20261206 each produced1,310,720 fresh own-game transitions. All41 native boundaries perseed, their frozen data, actual pause/resume, CUDA/CPU inference and latency checks passed. These infrastructure results do not establish learned chess strength.

| Seed | Final vs e8 score; adjusted98.75%CI | Paired SF512 score gain; adjusted98.75%CI | Final SF512 score | Strength |
|---|---|---|---|---|
|20261205|0.52604;[0.41146,0.64063]|+0.00521;[-0.05729,+0.06771]|0.06771|FAIL|
|20261206|0.51563;[0.41146,0.61979]|−0.01042;[-0.06771,+0.05208]|0.05729|FAIL|

All576 heldout games completed. An independent full-history audit verified56,835 legal plies,9,934 actual Stockfish move receipts and5,098,611 recorded engine nodes. Every arm had zero capped games. Each seed used48 distinct source-game roots, both colors perroot, and96 games perarm: final/e8, final/SF512, and frozen e8/SF512. Both candidates and e8 used the same16-simulation cached FullGumbel search and `max_considered_actions=4` schedule parameter. This parameter does not guarantee an actual four-child search cap. Stockfish19 used512 nominal nodes/turn, one thread/hash16/TT resets; observed node counts were separately audited.

The original gates remain unchanged: each seed needs direct score>.60 and adjusted lowerbound>.50; paired Stockfish gain>.10 and lowerbound>0; final SF512 score≥.25; each arm caps≤.05 plus adversarial-cap robustness; CPU masked inference latency ratio≤1.10. Fifty thousand source-root/color-paired bootstrap resamples used the frozen seeds and linear quantiles; MAX8 family multiplicity remains unchanged. These are conditional bootstrap intervals, not distribution-free guarantees or a general Elo estimate. Both seeds failed direct gain, paired SF gain, absolute SF performance and their relevant confidence bounds.

One original analysis assertion rejected a legitimate zero-Stockfish-move game: the candidate's first move ended the game. The separate versioned analysis accepts an empty node list only when exact legal replay proves zero opponent moves. It changes no statistical calculation, gate, game, model, book or seed. All old failures are retained.

- [Exact analysis and both seed results](../../experiments/ufuk/analysis-repair-v2/confirmation3/analysis.json), SHA `5c011fa884c35f0ca9762ee35748a1fd83460f48059ce246fb256f81298bbf99`.
- [Independent576-game full-history audit](../../experiments/ufuk/analysis-repair-v2/confirmation3/full576-audit-receipt.json), SHA `0bdf25dbd8fd45c739773290b107ab3a5bf321a28858e3ea94b492ba4e3870ca`.
- [Versioned analyser](../../experiments/ufuk/analysis-repair-v2/confirmation3/a100-mc-strength-analysis-v2.py), SHA `aadd71938956ba7bac7e535ea2b4d8255947545ebb6e035e5c62141769a015cb`.
- [Full native public Release persistence](UFUK-full-native-Release-persistence-result-20261005.json):84 archives/2,696,841,926 bytes, full anonymous size/SHA readback. This proves byte persistence; fresh CUDA restoration from the public archive has not yet been demonstrated.

Decision: continue with separately registered own-search acting and own-terminal curriculum mechanisms. Training loss, valid replay, improved engineering throughput, and minor unconfirmed point-score changes are not success. External teacher labels were not the continuous training target in this main40 attempt, but its measured self-learning strength objective failed. Apple Metal hardware remains untested; actual MLX CPU parity does not establish Apple GPU performance.
