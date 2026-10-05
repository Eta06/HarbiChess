import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import chess
import pytest
import torch

from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_ownsearch_learner import (
    TorchOwnSearchConfig,
    TorchOwnSearchLearner,
    tensor_bits_equal,
)

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = Path(
    os.environ.get(
        "HARBICHESS_TEST_E8",
        "/workspace/HarbiChess/artifacts/ufuk-a100-mirror-20261004/harbichess-inputs/initial-e8.safetensors",
    )
)
SOURCE = "d" * 40  # Synthetic source marker only; clean producer/CLI qualification belongs to root.


BASELINE_C93_SHA256 = {
    "harbichess/__init__.py": ("312219889131b7bb358c69edcdff60dc816d9244a460056e010b8c611c743225"),
    "harbichess/backends/action_value_network.py": (
        "2840f7301c77bc1007b125128271d3fc3ba1c6dfc8153441829a1ce1b8f13d02"
    ),
    "harbichess/backends/decoupled_value_network.py": (
        "8a09a4825f736f8dadb742cd3a0dca2b2c690f9feda567b0bdca3a0e66036d0f"
    ),
    "harbichess/backends/depth_transfer.py": (
        "cf4c633117facb334760fbf1d95e0f0583f953c607e63d37fd12f45394014966"
    ),
    "harbichess/backends/invariant_value_network.py": (
        "b241cf22f4344773fff20c7642c0ce3ffb5b5b9afc26deec5aef7e555540cb1b"
    ),
    "harbichess/backends/mlx_backend.py": (
        "a9f57980c5de6a5a4c4f4bccefcf07fa191ffb1be0c85d8a0cd4ae9030bb41c8"
    ),
    "harbichess/backends/mlx_context.py": (
        "f8a0863d60b8dd851dd527de6811ff15325896f593c1ed3f7f6f2aa893c3c19f"
    ),
    "harbichess/backends/mlx_network.py": (
        "506ffef7c3c4d55ef47567c6dd8cb5a8dd750f6df563c39f4c3a6a270c99300f"
    ),
    "harbichess/backends/mlx_smoke.py": (
        "93590a44f2afa22cb76ad15541115b0123ef13c38c9a7480589bdd60da1625ff"
    ),
    "harbichess/backends/mlx_sparse_value.py": (
        "4f25ba206e06500f1babe5d5c40d10b7fed4916f9e8c7f9988cc7f5192e807ac"
    ),
    "harbichess/backends/numpy_sparse_value.py": (
        "0f5877cda8b9545f3777960a349d9c814176464b8d69013e0cb6d53c52e31870"
    ),
    "harbichess/backends/pairwise_network.py": (
        "a4bbdb5630a9829705f6734710a39c7c982afcc6d19be4b1f7953773c3469a53"
    ),
    "harbichess/backends/pairwise_transfer.py": (
        "4257817bcc7918275ae90532a3806e7378fbfb0453aab8faa85edf38f4c653e1"
    ),
    "harbichess/backends/plastic_value_network.py": (
        "1eb6e42b910f6fcedcc876c6cc4d175ee4d392017c1cd6f35685cbef80ac7b94"
    ),
    "harbichess/backends/policy_adapter.py": (
        "58f2a1a8b09011ad4d24ff77cf4ccb42c1c604246e0b114cc54dac0be960e9d6"
    ),
    "harbichess/backends/policy_context_transfer.py": (
        "27214fdf2654f1e0683a0269ba06429f0ef0da312cac569d78a979d4c7b27300"
    ),
    "harbichess/backends/sparse_value_transfer.py": (
        "03855bac3915d406433aa5d3957e5725d9734c99839c46bd74c4f630866f0abc"
    ),
    "harbichess/backends/spatial_policy_network.py": (
        "4a8fd954494900124e029c035310f92dde28ccb2d3a86ecd279492884f76d51a"
    ),
    "harbichess/backends/torch_backend.py": (
        "d273bc5c2f22ccc227feb60768716551aacb8cff3c3279e3438cf8b906cfd5ba"
    ),
    "harbichess/backends/torch_context.py": (
        "0bb9381a1c2cf5bcec6e86292b045151049cf5191f9d51addedd376df4ebdaf4"
    ),
    "harbichess/backends/torch_network.py": (
        "42cf9b2a6840f7c3403b7e6ffd7a814c0ca11f9037a474abd1250048278a0644"
    ),
    "harbichess/backends/torch_sparse_value.py": (
        "51f14b79e46d7059ec6c59e3667c15bc7bc283b123161371091171cb5460f2e2"
    ),
    "harbichess/backends/ufuk_width_transfer.py": (
        "7ead8547ca9456c08ead4421a3e498d6a7943f58d932bf5c583fedc29f5d51a6"
    ),
    "harbichess/benchmarks/actors.py": (
        "a5962199f9b6288fe394d046d51f7c27acaa93ac370e4a0949bc40776874ad74"
    ),
    "harbichess/benchmarks/full_gumbel_batching.py": (
        "83417fae5e8f8595ad86cffd9425628def0387fd0ef382187be68b78332f0240"
    ),
    "harbichess/benchmarks/network.py": (
        "403a09b4f8345ff35aa8da9df00b07df970793a2b3e795bdc899ba2d9e8c9edf"
    ),
    "harbichess/benchmarks/oracle_search.py": (
        "e2e53b9386d82dcaf279c2aefcc5f0f3925b6331b7efd3011152bc603b4049c4"
    ),
    "harbichess/benchmarks/search.py": (
        "5d67bace9b3f8dff37667f8f70c6d921ae1df12f6322132837b4ff161fc71b78"
    ),
    "harbichess/benchmarks/torch.py": (
        "254176459458cf197cde4b88574283f31573f3e4b0e02cbb492bcc1e32ee6c3b"
    ),
    "harbichess/checkpoints/manifest.py": (
        "9a9df6d9f5905c2f81fccd0bef1b193d98660717532bec648178828a37978cff"
    ),
    "harbichess/checkpoints/release.py": (
        "f571a4574895070c67b418a21af3f099b4010554d36f3297cc9e4cc608e4fca6"
    ),
    "harbichess/chess/actions.py": (
        "84ac8a86a81002a2f4e00821b596450c5e8665e9f30735f77173cdf8e390e3cf"
    ),
    "harbichess/chess/encoding.py": (
        "d06cf156096b8ebadde2331ec406daed10f386a76fd7f9d4abfd63572a039ad9"
    ),
    "harbichess/chess/experimental_rule_cache.py": (
        "9d05e48e74669b6bca4d619a4eab5617c182d122f2010cd4f7a740edf0c52010"
    ),
    "harbichess/chess/packed_encoding.py": (
        "d63e57c277f6b79ee81b92b4fc84c6986b9d926becfa321413f6d779ab73f6a7"
    ),
    "harbichess/chess/rules.py": (
        "6843e3dff210c8a16e1faed831c7d3afb5da67e3bcf7c685820f54405537b5a0"
    ),
    "harbichess/core/backend.py": (
        "68910da9d2b6b5b3c9b43b314a482f5ca181e426b020651f9c0b4e4770509a88"
    ),
    "harbichess/core/network_config.py": (
        "677e061d20f3c19ef19b9295df85ef87412dd2837b03325cfc0a72648acaede2"
    ),
    "harbichess/core/state.py": (
        "736b546392194ac5ec07870d37b431e392570d06f1b93d71d77461a8f7fa844b"
    ),
    "harbichess/dashboard/server.py": (
        "b557b5ee2d3cf5aa899dc15ab57d676ad79b780a87154da0d7ea66f54985c3ce"
    ),
    "harbichess/dashboard/state.py": (
        "ded08450ff2901e9a582f23b82ea49d04e09d18927bb3b14161dc50063f7f0fb"
    ),
    "harbichess/evaluation/action_value_dataset.py": (
        "1b9a1a40660c111d41a787b334d556d995534d5dc425967a645e97cec194ddc0"
    ),
    "harbichess/evaluation/agreement_target.py": (
        "cfe33f05d759151ae6cb8729a79eb0efb66d59bf707d26fc23597a22f8fac2c2"
    ),
    "harbichess/evaluation/arena.py": (
        "586fd8b045379f7dae869fe1e3be1cf3f5ecd0e56a99867577d21a85129eb3fc"
    ),
    "harbichess/evaluation/consensus_target.py": (
        "a3ddc9966637ac06d3affda7128d9e3ac3b18e843398b33d4cc724ecab4432f5"
    ),
    "harbichess/evaluation/continuation_drift_audit.py": (
        "efe052b0e758022539479efd4136686edce965e86e425b290fee4a659d0cfcfd"
    ),
    "harbichess/evaluation/corrected_replay_value_transfer.py": (
        "197082f0fe072666743e45e20e47bdc9a89ca9922d11eb3f639c14799c3ae887"
    ),
    "harbichess/evaluation/cumulative_power_plan.py": (
        "f21772c78ff65312bdae6b5aced6feca3bcda1077a88cbaf9da3a3760ecc128b"
    ),
    "harbichess/evaluation/cumulative_value_gate.py": (
        "2798dd64e679910bd451f5859269f787bbe5b3ed2ac07f0cb607b5a68430efbe"
    ),
    "harbichess/evaluation/decisive_pair_teacher.py": (
        "afdc89803a97c19a33a69f17dfc2510fb6cffe9ed1ccf8d0d69d6860517dc6c4"
    ),
    "harbichess/evaluation/decoupled_value_qualification.py": (
        "dc27f396051c7ff3c1912d8c35171ee0c4ed1fef5d53ed27c841b35b76f24182"
    ),
    "harbichess/evaluation/deterministic_value_probe.py": (
        "1a7f7038772126516886126c383b622d10ece52c5ccf7fe7c7f5579c0f684eff"
    ),
    "harbichess/evaluation/full_gumbel_targets.py": (
        "bb9796af219884fd1a2e1fdb8f8ca0c52ba2aa1717ff3a57807eced8bd904391"
    ),
    "harbichess/evaluation/invariant_value_probe.py": (
        "95eede0dce494ee47effe4b1e4f407b1e423781c32ede0f49fe263b849bfd2ba"
    ),
    "harbichess/evaluation/model_quality.py": (
        "029732cf418845e98246496aab9696497099518660def33064d588cb278bbe0a"
    ),
    "harbichess/evaluation/paired_comparison.py": (
        "0272893a5933b559325dc220a9d923d42de781687a1ca60848a1b88d2be56b63"
    ),
    "harbichess/evaluation/paired_gate.py": (
        "8379aef64b27281f4d5f6199b81fdb719e5d5d99f106c248a17cdcbd8909f08c"
    ),
    "harbichess/evaluation/policy_candidate_validation.py": (
        "117de9b12fbffda8d080d603a0da83ea1ff3715fc7dda14abe3f273d79102db5"
    ),
    "harbichess/evaluation/policy_improvement_target.py": (
        "86459682262f002fe31b2a4c5c2e79608e93f29f6eea2ad555623a91e13aee6b"
    ),
    "harbichess/evaluation/policy_signal.py": (
        "838303bad4bcfff4900b6139d22cc564c561dc7f0df9c7ee1fc2a2aea1165b47"
    ),
    "harbichess/evaluation/portable_arena.py": (
        "7d42799b1a6ed8e4363982b6803bb755ced2e64966460edc2ae93e6575896e60"
    ),
    "harbichess/evaluation/quality.py": (
        "96d60960029dde7bfac840bbe33e425c9ff6b24fb2d2a119e6ebfd6e6402fcb3"
    ),
    "harbichess/evaluation/replay_policy_signal.py": (
        "f1444dd62b1ed3d44a885bff1ff457bf67e7b815dede9a461ba2fb26ecf5f55a"
    ),
    "harbichess/evaluation/replay_teacher_alignment.py": (
        "1cb9ddde2e1964ef0b74214eaa3b56b48346da5c07b89413fe09ce6525cca12f"
    ),
    "harbichess/evaluation/root_budget.py": (
        "684a7872d17a63b1395fa9f06ce64375fb52c7b3beb28a7b7af1c5f5ccf03ba4"
    ),
    "harbichess/evaluation/search_diagnostics.py": (
        "eab6edeccbef65b11f39bcb3266b7acdad920d42a71e506254425b3f4d8b734b"
    ),
    "harbichess/evaluation/search_q_reliability.py": (
        "b7238b92792ecfe761fd29ce9a3f5cac94798a256ed31b5298e17db011da2f61"
    ),
    "harbichess/evaluation/sequential_halving_qualification.py": (
        "2ffa443290459cce800d329b7d297fda3593db03bb4e9fcf72df3a7531aa5b4d"
    ),
    "harbichess/evaluation/system_teacher_qualification.py": (
        "34fb3d67052c0a0cfdbb814690c22b6884c2ba5aece6b650b0c092ca1ff1ff3b"
    ),
    "harbichess/evaluation/teacher_consistency.py": (
        "bab19cc3989436bd1422a3ddb5d7268dba3c940cd3b63bb88a1ff676c4a079b2"
    ),
    "harbichess/evaluation/teacher_instability.py": (
        "bb8eef7015aff4e41a92894b855917331943b3ea7704e105f6dd8e1565176718"
    ),
    "harbichess/evaluation/teacher_probe.py": (
        "65a1086012e398ccb31ffa37ece6e73ddcf4232f5212169289e237d8d5341d71"
    ),
    "harbichess/evaluation/teacher_qualification.py": (
        "a7dca2a806a0e08e865e90d28dccb0a43c9a1d82d371ffb360b48f5bf87435b1"
    ),
    "harbichess/evaluation/uncertainty_q_labels.py": (
        "567857864929a6a885d0d51d2c7dfc67017a469c40dfd08b5e1f65c30b3eccd8"
    ),
    "harbichess/evaluation/value_calibration_ablation.py": (
        "45a23b8e501cd50fc04c7d7f99a41755b3f96b0a29d39b65d9c73a01ddeeb086"
    ),
    "harbichess/evaluation/value_oracle_diagnostics.py": (
        "cf6e463b7fd3fd1eb0d0dad3ebe53a18e015f15be0be29637ab67d50b8ac0482"
    ),
    "harbichess/evaluation/value_pipeline_diagnostics.py": (
        "7bb8767d21598ae6016018f3f2e0614282c5f07e78ce5b93ff4a42673f53137d"
    ),
    "harbichess/evaluation/value_signal_audit.py": (
        "27831a4ec28e11b8873a4a52588761c8dfee54b97bc5b0b9bd25828c2044b4bc"
    ),
    "harbichess/evaluation/value_target_conflict.py": (
        "b10aa6735419e79b8befd505f6ed3fa8691378e5a45b08d754ade14ebc16728e"
    ),
    "harbichess/replay/audit.py": (
        "5c48663f1e831ce0ad5b49d418674d87d89d44e49288314e43089b5e50423e64"
    ),
    "harbichess/replay/branch_evidence.py": (
        "b9706a84a7008bbc37a1bb13cd1fcff423f9083c589abb34dfac51cb087da27d"
    ),
    "harbichess/replay/coverage.py": (
        "3831629cdafeceef827a077520f8ff9790c7235a38675d22c6d6eb5d799c23d4"
    ),
    "harbichess/replay/diversity.py": (
        "73094f6d630433bf143a27c1e607fdf69c054df472b61d3373de693b72c1eb99"
    ),
    "harbichess/replay/merge.py": (
        "af8fe156459706fe31ab6b436b6f28b5327fd661ce08f171b33a931b4f5ff10c"
    ),
    "harbichess/replay/repetition_risk.py": (
        "fc5d5228e867ef51f165941bb0e44a069cdf9db23b0efbcd99c2df49ad184251"
    ),
    "harbichess/replay/schema.py": (
        "2776f66a1379469ccc8d081c47a4abdf2762ba040ba17065d27c7b5dacdb6da9"
    ),
    "harbichess/replay/shard.py": (
        "a9f83f0d840c04021344d36843084ad27212fcb5015350f624270a9648da73fc"
    ),
    "harbichess/replay/split.py": (
        "44576d9c7d40c69ab066d3ecc025ad426f7dfb0fc959f04a91488373b1f94ff9"
    ),
    "harbichess/replay/value_aware_risk.py": (
        "27f19a97d69304a37d98d4318cac7eb365d7b59c254457314934b99f5b526361"
    ),
    "harbichess/replay/value_regret.py": (
        "419944fb8bff194336d8c5554882237e80a7ec0144dc9cf487f754e650b46d7f"
    ),
    "harbichess/search/all_legal.py": (
        "c5cc7c0c48ff733b9d223f1afab0ac4b12c6d06e9467341043eaeb0420b904c5"
    ),
    "harbichess/search/batching.py": (
        "0c5da73222921ca226b9d7735bb2814f5ef602d4ec2c18fe154ef14fbe181d46"
    ),
    "harbichess/search/continuation.py": (
        "6c38581f8b376f552a56609f790038682f28f18795358fe0e6db47eb5e2cbcfd"
    ),
    "harbichess/search/diagnostics.py": (
        "3f90d3997c8ca37e056c5fc1022c1ed358030e003fdb6a7b99231e88b7c29e84"
    ),
    "harbichess/search/evaluator.py": (
        "c656c57f75eec6745ac8edd88df59c0d1c1560be4288490fab838c4be7d308f4"
    ),
    "harbichess/search/full_gumbel.py": (
        "91c0f64adb29b38019c61966632fa7099708037d488643447570afb65abb79b3"
    ),
    "harbichess/search/gumbel.py": (
        "74a449b32cfe5fc0ca9c64f6ad4b09312947aac8ddfac0faca935a591c0ac850"
    ),
    "harbichess/search/mcts.py": (
        "e3e79d874f1822fef0362f6b7efe250ff06050eda68d4bce8c6039bf7da35e8d"
    ),
    "harbichess/search/ownsearch_wavefront.py": (
        "9bd73661db1453271fa322d3c3e43141f4a37a2bc4f88e2c1b14e4301e921d67"
    ),
    "harbichess/search/q_range_floor.py": (
        "cdff19de7171fcfd37f9ba2379e0403f55bf13a19fdc941ad95688bc12ed9089"
    ),
    "harbichess/search/root_halving.py": (
        "d65f58c2bb665c6f1e270322a2400abcf2bcfe25210bf354cba14d12c70c845a"
    ),
    "harbichess/search/sequential_halving.py": (
        "9162d7027ea3ef78e177382f690f212af667487973e4c9ff4685d77fb629303a"
    ),
    "harbichess/search/tactical_leaf.py": (
        "9840037e58c9ec47e065da2453b1b44f48958ee50859b32ce2eb1772fd1ff2de"
    ),
    "harbichess/search/targets.py": (
        "dd33da1dc12fd14e06b3fc0b46ed12e271909a7fb6376d819eebd255ebd8d429"
    ),
    "harbichess/search/value_oracle.py": (
        "88c199a35d007a76a02a3ca0efc98d8792798e42de1292625dfed61119394ade"
    ),
    "harbichess/search/value_policy.py": (
        "c7000980bbb6bc72a465b97533f10b5bfe44a26b0418705c697ccb471a3ab39c"
    ),
    "harbichess/selfplay/continuous_replay.py": (
        "71b4230932d752aef7e71786a276589497bada92ae65c28dfd66a75732fbf1ff"
    ),
    "harbichess/selfplay/game.py": (
        "ba04b95b7fec91af7b92c2c894886ec5ba9b8d0d564776b195729069a35e0875"
    ),
    "harbichess/selfplay/online_actor.py": (
        "b544082b72b9771f87a4bf09f5f8c1613887dcbbf8ae76531a4efd12532f0017"
    ),
    "harbichess/selfplay/online_epoch.py": (
        "748052d73cf01dc560dc7432514b54cc63b194f59d366dd66b8df6f6acc6d162"
    ),
    "harbichess/selfplay/parallel.py": (
        "3fb5bc816a415707f201674f948fe1b3fdf7aa37b30bca78a3e486f1332ae2c2"
    ),
    "harbichess/selfplay/torch_actors.py": (
        "87637dbc6506355b0f02faa56058d4b26db2b4a1b1edaaadebb208ac18c6e57d"
    ),
    "harbichess/training/ablation.py": (
        "a56dcdf402397101bee002fd9fc28110c37a544105086fcdd494113b21058da8"
    ),
    "harbichess/training/action_value_transfer.py": (
        "71d028fb2eeb8d6197522009b07e5c5fb4985f0c4effb8505f5f7cbb4384f0fa"
    ),
    "harbichess/training/anchored_policy_transfer.py": (
        "3ecb904c2863a60f014db5ab400e05f798df1cc6f58d2e3569498549ed7ac531"
    ),
    "harbichess/training/batch.py": (
        "412ada4777dd3b1e12b10e65e5d591cf61ace8b2d4cc2000942ee0e6c81cb9e4"
    ),
    "harbichess/training/capacity_matrix.py": (
        "39f01c6d93fdecb733ef69d02512d517fd1d7b5958d8389db1ff99ec6fd6faa3"
    ),
    "harbichess/training/checkpoint.py": (
        "2045a0d2abf56814153e19004f17981a990da6455e5053eef44ba95d4fdfa29b"
    ),
    "harbichess/training/config.py": (
        "2526e0842c51f1afdaf6dccbd534494f986f8337a11bcb50ddccbe7fdce2d01f"
    ),
    "harbichess/training/constrained_plastic_ablation.py": (
        "1df6ba9cc7544e8d7d4c7d1145d08b19f0bdd4db2709d07474ddd8e3d0a31a40"
    ),
    "harbichess/training/continuous_checkpoint.py": (
        "4bd2ae4e7ffd3af6cc6d5a39851f0744067299a4d977f8c44da7cde2155a95d2"
    ),
    "harbichess/training/continuous_policy_iteration.py": (
        "9c8f37894f4dc8a750681e3a945a5182867b4a41a5aea557dbaf8dff9d038cb2"
    ),
    "harbichess/training/controlled_oracle_train.py": (
        "534734dd1303e0f691153cfffdc37a27722fe8c7b4e39b3a390068c2a0a71080"
    ),
    "harbichess/training/decoupled_value_transfer.py": (
        "406c3a26fc928c9ac22b434f7450b39f1a08e391caffb39cbf30c1cb167f33df"
    ),
    "harbichess/training/full_gumbel_anchor_ablation.py": (
        "6d44ce48a7f27dd6664968ebbfe190432956d99046ab433b036eb71c71c1168a"
    ),
    "harbichess/training/full_gumbel_transfer.py": (
        "c27da2331e24168b846ac64fa479f5cbfdaca5f298691e1095ce57a9610cc713"
    ),
    "harbichess/training/fullgame_own_targets.py": (
        "4592305598a13b89e1cae3b13e66ca08159e4ede139265cbc214b828143049dd"
    ),
    "harbichess/training/history_openings.py": (
        "bea86f921dbbdc01bc007b9735d259309c83395daade2c43f4a4e7064e490c38"
    ),
    "harbichess/training/invariant_wdl_transfer.py": (
        "7641b722449b7101e1e25a11d2f27c3d59df3576e7c0bb7c38b53b3c19ff2eac"
    ),
    "harbichess/training/joint_policy_transfer.py": (
        "896b60d1993af24d4fa365a2083911237622c3d60ba5242d7fa7ecd6e54e046f"
    ),
    "harbichess/training/joint_policy_value_transfer.py": (
        "d722f67446c96877e0e9d51ec7fd5b51550a7b3676bc48b3c75216daeb3b4ccc"
    ),
    "harbichess/training/learner.py": (
        "88067d0a8038d312050f7be3f16764ccffb0b8292eb0dc7d1fb45b873b7878c5"
    ),
    "harbichess/training/learner_transfer.py": (
        "77bfbfd74a6fe8d15df4fe06e5a7ebe307c3f991a6e8974ee93ba643823ab58a"
    ),
    "harbichess/training/merge_oracle.py": (
        "af537a3242b475f9aec4ecc8fe8e510383da321958f5b1c69d294aef6969570e"
    ),
    "harbichess/training/mlx_online_objective.py": (
        "b017a08410366e67ea3bb7a5217eba5f24f841b07663364aad0fb7bb165b7a15"
    ),
    "harbichess/training/multistep_own_targets.py": (
        "39c92edb60a68ea841648591a294ab18e7827e39f0ad50437f62e126f8b88dd6"
    ),
    "harbichess/training/network_expansion.py": (
        "ee82420ce554a8b5a4bbdebf02edbb9db07fb1cc5fd5facc4d61fe9d3b45e6af"
    ),
    "harbichess/training/ocak_run.py": (
        "3e1f56034bc82bb7b8432e46f472d9daaea80dd2d035a2f343bba5dd2a5eab27"
    ),
    "harbichess/training/online_objective.py": (
        "8a1f7e5c07dcdd495dba2895aae98a5626bb12a37ef85b73758d74cb12ce542e"
    ),
    "harbichess/training/online_targets.py": (
        "0994408957f7c96da60f8621b46afd624c821d0aad8f53888c90ac02292d1230"
    ),
    "harbichess/training/oracle_data.py": (
        "15a1b1dcb42eedb52bd93a1e441b33782814696bc6614bbd88f252d1e7657199"
    ),
    "harbichess/training/oracle_train.py": (
        "25a821019b92564c5d31d8bd65abddd542a8d4173cb339bfac56eeac8544c5ec"
    ),
    "harbichess/training/ownsearch_targets.py": (
        "1c4b0bf33bd6b89b058fb316ebf4cb53c2bebfe0cb2f134cdf1ff77a250bea36"
    ),
    "harbichess/training/pilot.py": (
        "1474e03638424ba7be61cea101524bfa767eaf33adeb20da6e1ebf47d8e878c1"
    ),
    "harbichess/training/policy_convergence.py": (
        "8ddcda6dfcc6c8b1eb0102acc0bbb48e8a91a3ce3fcf5cb4a0afadb84cdb0035"
    ),
    "harbichess/training/policy_projection.py": (
        "539ba6388dc49b4a607d18a53f1d81440023999c72584ee678efae29a553aa76"
    ),
    "harbichess/training/prepared_oracle.py": (
        "47ee47221a7e37f8c41a7893f982dc88de6b80b9debf7909e4c8c9bc20f97fbe"
    ),
    "harbichess/training/prepared_oracle_torch.py": (
        "2d201b44a635db17626d8e1c00ee24cac3cbfec2679470bab61189a6f6d34c6e"
    ),
    "harbichess/training/replay_distill.py": (
        "8bc2aafdfd12d946720b8742c68bf18b218b722066ec3ccdac6472d79c76131e"
    ),
    "harbichess/training/resume.py": (
        "ce9ecfd194e20ec06d15a852790fbd32988b05ff74aecdce83560c2f636d5e5e"
    ),
    "harbichess/training/short_horizon_value.py": (
        "f9467762b9dd57bd59e25ccb0490be8a44ad19691a6f784fc2ba839cf3aee29f"
    ),
    "harbichess/training/soft_wdl_ablation.py": (
        "9d02ca056de2a5a0d91eb98ae4731912431fc65c01df692005f65e376d7ae07f"
    ),
    "harbichess/training/spatial_action_value_transfer.py": (
        "cf9248a1c800c763687f7ea98032bb7b7d31a0fa2f7fcae9c97e1b9f0f51d315"
    ),
    "harbichess/training/spatial_policy_transfer.py": (
        "e76ec97922ef2e1dcb604c24c69c112ae67fd3e6b30631c9023c08575fca48b3"
    ),
    "harbichess/training/stable_plastic_ablation.py": (
        "7bf568d95a8ca971c7c0cc6821ae839e12622c2b727c5d18464db0e67f02dd12"
    ),
    "harbichess/training/torch_checkpoint.py": (
        "2621b0919ed90072c696d4e1c5ceda046fb8d191c3cbfacfc5312588ab901c9a"
    ),
    "harbichess/training/torch_fullgame_checkpoint.py": (
        "399f3fb533bc81a410d6887263f02bd9a3afea5d5628d23416ddcfabd42b1aee"
    ),
    "harbichess/training/torch_fullgame_learner.py": (
        "b38d93e9b5b19801f3ba13ec7ac33f0e7736691a2ad287937f46abc0a377d6b0"
    ),
    "harbichess/training/torch_fullgame_ppo.py": (
        "e864505c3cef3c96ae07e508b329c39cac65fd1ac2072556aa8e80ee8f8611c4"
    ),
    "harbichess/training/torch_fullgame_run.py": (
        "2ef67ad43df89e1c55924dce60e1a260933c74b066af9b6be8ce30a9a4f7a23c"
    ),
    "harbichess/training/torch_learner.py": (
        "429bff785de91ad3316096eb1f6de64d0783d592acbc2a0b5be77061999d7673"
    ),
    "harbichess/training/torch_loop.py": (
        "693c7c41d757738f2c24447a70c2bcec7aad8f83ad68ff955ff1af31b7e777cb"
    ),
    "harbichess/training/torch_online_checkpoint.py": (
        "c707d2a195fc50aabdfc4d82bb1d50b4ab50308a20e2f542d30b601a4976c8c7"
    ),
    "harbichess/training/torch_online_learner.py": (
        "4832423c80f2397c5a8e31856015c342aaddcb9f452f25c2fd9ea10abd275d2f"
    ),
    "harbichess/training/torch_online_objective.py": (
        "092c4c7708f6b5c0271f182378cf3c46db16e92cbf924c4cc0603267cb641443"
    ),
    "harbichess/training/torch_online_run.py": (
        "cd6aaa470e68f0f73937dc6cd288a0ca3a84f787699ed28653e859544b6808ff"
    ),
    "harbichess/training/torch_ownsearch_checkpoint.py": (
        "6169b535d0bec0979505ec263d204e24568daeb158f6f504973ef89d8eb0f529"
    ),
    "harbichess/training/torch_ownsearch_core.py": (
        "f8606f2fe2135d00d17790ef0bda2e5b04e9e5a3e31e32f9d3f6866132fb8ad8"
    ),
    "harbichess/training/torch_ownsearch_learner.py": (
        "ccb11452f7ce9b6400da95a81b1bf292de15b8cb786a71f67f2008457e1c23ac"
    ),
    "harbichess/training/torch_ownsearch_run.py": (
        "53dddac03263dde3deb37051f8cb3496a6be15a92233cd335387909dc528e9df"
    ),
    "harbichess/training/ufuk_context_guard.py": (
        "ec248b04c80cd76eb7bfb6871f558e095221e7171494813bcc4177fe51583192"
    ),
    "harbichess/training/ufuk_joint_context_guard.py": (
        "2279d99f9ea305560b127fed95c0241c9ca35c85e1744472fa21406abbf02e00"
    ),
    "harbichess/training/ufuk_sparse_guard.py": (
        "9dc6731e80221a0eda7eb44f12c5c46168ffbfe531666aada790ced2f001fc35"
    ),
    "harbichess/training/ufuk_value_outcomes.py": (
        "d2530fa8ea1ab67a67d401f37541f8664f8703d24d2b6c4c7c1d48fe630f4542"
    ),
    "harbichess/training/ufuk_width_guard.py": (
        "a693cd7a917bca9e62ceea23403cfd02a4bb87d6bf3a0af0b8add0c0135f097e"
    ),
    "harbichess/training/uncertainty_policy_transfer.py": (
        "6684e760e0d638409b7957f196510a3f942279036d14200e30336855c0a9d44c"
    ),
    "harbichess/training/value_bootstrap.py": (
        "57a95b00295e4e1b61e0cbd73463db1a8a6b0849e6afa8ee07ab43200c94ba45"
    ),
    "harbichess/training/value_calibration.py": (
        "e5f03afd880d6ceda1b7280b5116fe2d36918903c5538d1025479def7d9631cb"
    ),
}


def fixture(tmp_path):
    data = dict(
        seed=20261405,
        actors=dict(games=4, max_additional_plies=8, claim_draw=True, temperature=1.0),
        objective=dict(
            policy_weight=1.0,
            value_weight=0.2,
            policy_anchor_weight=0.03,
            value_anchor_weight=0.02,
            behavior_kl_stop=1.0,
        ),
        search=dict(
            simulations=16,
            max_considered_actions=4,
            gumbel_scale=0.0,
            value_scale=0.1,
            maxvisit_init=50.0,
            block_plies=8,
        ),
        schedule=dict(minibatch_size=4, passes=2, max_gradient_norm=5.0),
        epoch_steps=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device="cpu",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data) + "\n")
    book = tmp_path / "book.json"
    book.write_text(
        json.dumps(
            dict(
                schema=1,
                splits=dict(
                    train=[
                        dict(
                            source_game="white",
                            root_ply=0,
                            opening=dict(root_fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", moves=[]),
                        ),
                        dict(
                            source_game="black",
                            root_ply=0,
                            opening=dict(root_fen="8/8/8/8/8/6k1/5q2/7K b - - 0 1", moves=[]),
                        ),
                        dict(
                            source_game="broad",
                            root_ply=0,
                            opening=dict(root_fen=chess.STARTING_FEN, moves=[]),
                        ),
                    ]
                ),
            )
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"scope":"local CPU unit infrastructure only, not production"}\n')
    return config_path, dict(
        initial_weights=WEIGHTS,
        book=book,
        experiment_config=config_path,
        protocol=protocol,
    )


def config(path):
    data = json.loads(path.read_text())
    data["actors"] = OnlineActorConfig(**data["actors"])
    data["objective"] = OwnSearchObjective(**data["objective"])
    data["search"] = OwnSearchConfig(**data["search"])
    data["schedule"] = FullGamePPOTrainConfig(**data["schedule"])
    return TorchOwnSearchConfig(**data)


PROCESS = r"""
import json,sys,torch
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_ownsearch_learner import TorchOwnSearchConfig,TorchOwnSearchLearner
root,cfg,weights,source,mode=sys.argv[1:];root=Path(root);cfg=Path(cfg)
c=json.loads(cfg.read_text());c['actors']=OnlineActorConfig(**c['actors']);c['objective']=OwnSearchObjective(**c['objective']);c['search']=OwnSearchConfig(**c['search']);c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
inputs=dict(initial_weights=Path(weights),book=cfg.parent/'book.json',experiment_config=cfg,protocol=cfg.parent/'protocol.json')
torch.set_num_threads(1)
l=(
 TorchOwnSearchLearner.resume(
  root/'native1',config=TorchOwnSearchConfig(**c),input_paths=inputs,source_commit=source
 ) if mode=='resume' else TorchOwnSearchLearner.fresh(
  config=TorchOwnSearchConfig(**c),input_paths=inputs,source_commit=source
 )
)
assert l.actors.rules.board_cache_size == 8192
root.mkdir(exist_ok=True)
while l.epoch<(1 if mode=='pause' else 2):
 l.train_epoch();(root/f'journal{l.epoch}.gz').write_bytes(l.last_epoch_gzip);l.checkpoint(root/f'native{l.epoch}')
print(json.dumps(dict(epoch=l.epoch,accepted=l.optimizer_accepted_updates,search_roots=json.loads(__import__('gzip').decompress(l.last_epoch_gzip))['training']['policy_target_rows'])))
"""


def test_only_ownsearch_fresh_and_resume_before_validation_use8192(tmp_path, monkeypatch):
    import harbichess.training.torch_ownsearch_checkpoint as checkpoint
    from harbichess.chess.rules import PythonChessRules
    from harbichess.training.torch_ownsearch_learner import OWNSEARCH_BOARD_CACHE_SIZE

    torch.set_num_threads(1)
    assert OWNSEARCH_BOARD_CACHE_SIZE == 8192
    assert PythonChessRules().board_cache_size == 512
    path, inputs = fixture(tmp_path)
    learner = TorchOwnSearchLearner.fresh(
        config=config(path), input_paths=inputs, source_commit=SOURCE
    )
    assert learner.actors.rules.board_cache_size == 8192
    learner.train_epoch()
    learner.checkpoint(tmp_path / "native1")
    original = checkpoint.build_fullgame_targets
    calls = []

    def guarded_validation(rules, *args, **kwargs):
        assert rules.board_cache_size == 8192
        calls.append(True)
        return original(rules, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "build_fullgame_targets", guarded_validation)
    resumed = TorchOwnSearchLearner.resume(
        tmp_path / "native1",
        config=config(path),
        input_paths=inputs,
        source_commit=SOURCE,
    )
    assert calls and resumed.actors.rules.board_cache_size == 8192
    assert PythonChessRules().board_cache_size == 512


def test_full_gumbel_e8_matches_source428_on_18_full_histories_and_draw_modes(tmp_path):
    """The certificate treatment leaves the registered FullGumbel evaluator unchanged."""
    import json
    import os
    import random
    import subprocess
    import sys

    source428 = Path(os.environ.get("HARBICHESS_TEST_SOURCE428_SRC", str(HERE / "baseline428/src")))
    candidate = HERE / "src"
    assert source428.is_dir() and candidate.is_dir()
    roots = []
    # One exact threefold-claim root exercises the configured claim_draw branch.
    roots.append(
        dict(
            root_fen=chess.STARTING_FEN,
            moves=["g1f3", "g8f6", "f3g1", "f6g8"] * 2,
        )
    )
    # Seventeen distinct, legal, nonterminal training-style full histories.
    seen = {tuple(roots[0]["moves"])}
    for index in range(17):
        rng = random.Random(20261005 + index)
        board = chess.Board()
        moves = []
        for _ in range(6 + index):
            if board.outcome(claim_draw=False) is not None:
                break
            move = rng.choice(tuple(board.legal_moves))
            moves.append(move.uci())
            board.push(move)
        assert board.outcome(claim_draw=False) is None
        assert tuple(moves) not in seen
        seen.add(tuple(moves))
        roots.append(dict(root_fen=chess.STARTING_FEN, moves=moves))
    assert len(roots) == 18
    states_path = tmp_path / "full-history-roots.json"
    states_path.write_text(json.dumps(roots, sort_keys=True) + "\n")
    script = r"""
import json, random, sys, torch
from pathlib import Path
from harbichess.backends.torch_network import load_weights
from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
torch.set_num_threads(1)
rules=PythonChessRules()
backend=TorchPolicyValueBackend(load_weights(Path(sys.argv[2])).eval(), device='cpu')
class SinglePositionBackend:
    def evaluate(self, position):
        return backend.evaluate([position])[0]
evaluator=NeuralPositionEvaluator(SinglePositionBackend(), rules=rules)
roots=json.loads(Path(sys.argv[1]).read_text())
outputs=[]
for item in roots:
    state=ChessState(item['root_fen'], tuple(ChessMove(move) for move in item['moves']))
    for claim in (True, False):
        result=FullGumbelMCTS(
            evaluator, rules=rules,
            config=FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, claim),
        ).search(state, rng=random.Random(8128))
        outputs.append({
            'root_fen': item['root_fen'], 'moves': item['moves'], 'claim_draw': claim,
            'simulations': result.simulations, 'root_value': result.root_value,
            'outcome': None if result.outcome is None else result.outcome.termination,
            'selected_action': (
                None if result.selected_action is None else result.selected_action.uci
            ),
            'moves_stats': [[x.move.uci, x.visits, x.prior, x.mean_value] for x in result.moves],
            'policy': [[m.uci, p] for m, p in result.action_weights],
        })
print(json.dumps(outputs, sort_keys=True, separators=(',', ':')))
"""
    outcomes = []
    for label, source in (("source428", source428), ("certificate", candidate)):
        env = dict(
            os.environ,
            PYTHONPATH=str(source),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(states_path),
                str(WEIGHTS),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{label}: {result.stderr}"
        outcomes.append(json.loads(result.stdout))
    assert outcomes[0] == outcomes[1]
    claimable = outcomes[0][0]
    assert claimable["claim_draw"] is True and claimable["simulations"] == 0
    unclaimed = outcomes[0][1]
    assert unclaimed["claim_draw"] is False and unclaimed["simulations"] == 16


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="actual CUDA unavailable; A100 zero-skip cache proof required",
            ),
        ),
    ],
)
def test_old_and_cache_native_journals_all_rng_exact_and_fresh_resume(tmp_path, device):
    import gzip

    from harbichess.backends.torch_network import load_weights

    baseline_src = Path(os.environ.get("HARBICHESS_TEST_C93_SRC", str(HERE / "baseline/src")))
    source428 = Path(os.environ.get("HARBICHESS_TEST_SOURCE428_SRC", str(HERE / "baseline428/src")))
    # Source428 is the matched no-certificate parent. Only these explicitly
    # versioned files may differ in the candidate treatment.
    treatment_files = {
        "harbichess/search/full_gumbel.py": (
            "b90cb242cae639f66f7fe4273ddff47694069c1097b1af88a639cb07b4ccc327"
        ),
        "harbichess/search/ownsearch_wavefront.py": (
            "d2c9f5e8259ae48b29249d504ca8411da0e4b9b2ac03287e14310aaf4f6a9714"
        ),
        "harbichess/training/ownsearch_targets.py": (
            "c9e7273238ccaa2b4bbe336b7780c8e1c19e70af792040f2967e78689335438d"
        ),
        "harbichess/training/search_acting_epoch.py": (
            "0837bc1d41840f40f35b6c0d19ff9df3300b8adc6b6e1c2f898040e01b0f33a8"
        ),
        "harbichess/training/torch_ownsearch_checkpoint.py": (
            "dee14a5a4aadd1dbbff954168bcf0b3b02df47fb3144c895e877ad7ef9ce3430"
        ),
        "harbichess/training/torch_ownsearch_learner.py": (
            "3f529db9f573caab26f53bf626327070c7db87b87740ad4c489c6b95bf1388a8"
        ),
    }
    assert source428.is_dir()
    for relative, digest in BASELINE_C93_SHA256.items():
        assert hashlib.sha256((baseline_src / relative).read_bytes()).hexdigest() == digest, (
            relative
        )
        candidate_file = HERE / "src" / relative
        if relative in treatment_files:
            assert (
                hashlib.sha256(candidate_file.read_bytes()).hexdigest() == treatment_files[relative]
            ), relative
        else:
            assert candidate_file.read_bytes() == (source428 / relative).read_bytes(), relative

    path, _ = fixture(tmp_path)
    raw = json.loads(path.read_text())
    # No root in this deliberately short fixture has a one-ply mate. This
    # isolates array/cache/native behavior from the certificate treatment.
    raw["actors"]["max_additional_plies"] = 1
    raw["epoch_steps"] = 1
    raw["search"]["block_plies"] = 1
    raw["device"] = device
    path.write_text(json.dumps(raw) + "\n")
    book_path = tmp_path / "book.json"
    book = json.loads(book_path.read_text())
    for row in book["splits"]["train"]:
        row["opening"] = dict(root_fen=chess.STARTING_FEN, moves=[])
    book_path.write_text(json.dumps(book, sort_keys=True) + "\n")
    for label, mode, candidate in [
        ("parent428", "whole", False),
        ("candidate", "whole", True),
        ("split", "pause", True),
        ("split", "resume", True),
        ("parent428", "audit", False),
        ("candidate", "audit", True),
        ("split", "audit", True),
    ]:
        root = tmp_path / label
        source = SOURCE if candidate else "8" * 40
        code = PROCESS
        if mode == "audit":
            code = code.replace("root/'native1'", "root/'native2'").replace(
                "mode=='resume'", "mode=='audit'"
            )
        env = dict(
            os.environ,
            PYTHONPATH=str(HERE / "src" if candidate else source428),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(root),
                str(path),
                str(WEIGHTS),
                source,
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
    parent = tmp_path / "parent428/native2"
    candidate_native = tmp_path / "candidate/native2"
    split = tmp_path / "split/native2"
    parent_training = torch.load(parent / "training.pt", map_location="cpu", weights_only=True)
    candidate_training = torch.load(
        candidate_native / "training.pt", map_location="cpu", weights_only=True
    )

    def assert_tree_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and torch.equal(left, right)
        elif isinstance(left, dict):
            assert isinstance(right, dict) and left.keys() == right.keys()
            for key in left:
                if key == "sample_chain_sha256":
                    assert len(left[key]) == len(right[key]) == 64
                    continue  # Each versioned record chain hashes its own format.
                assert_tree_equal(left[key], right[key])
        elif isinstance(left, tuple | list):
            assert type(left) is type(right) and len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                assert_tree_equal(a, b)
        else:
            assert left == right

    # All trainable, optimizer, actor, sampler and global RNG state must match
    # on this certificate-inactive trajectory, despite the versioned journal.
    assert_tree_equal(parent_training, candidate_training)
    assert_tree_equal(
        candidate_training,
        torch.load(split / "training.pt", map_location="cpu", weights_only=True),
    )
    parent_actor = json.loads((parent / "actor.json").read_text())
    candidate_actor = json.loads((candidate_native / "actor.json").read_text())
    parent_actor.pop("sample_chain_sha256", None)
    candidate_actor.pop("sample_chain_sha256", None)
    assert parent_actor == candidate_actor
    parent_record = json.loads(gzip.decompress((parent / "last-frozen-epoch.json.gz").read_bytes()))
    candidate_record = json.loads(
        gzip.decompress((candidate_native / "last-frozen-epoch.json.gz").read_bytes())
    )
    split_record = json.loads(gzip.decompress((split / "last-frozen-epoch.json.gz").read_bytes()))
    assert parent_record["own_search"]["schema"] == "ownsearch-random-block-targets-v1"
    assert candidate_record["own_search"]["schema"] == "ownsearch-random-block-targets-v2"
    assert parent_record["own_search"]["targets"] != candidate_record["own_search"]["targets"]
    for record in (candidate_record, split_record):
        assert all(root["certified_mates"] == [] for root in record["own_search"]["roots"])
    for record in (parent_record, candidate_record, split_record):
        record.pop("sample_chain_sha256", None)
        record.pop("previous_sample_chain_sha256", None)
        record["own_search"].pop("schema", None)
        record["own_search"].pop("targets", None)
        for root in record["own_search"]["roots"]:
            root.pop("certified_mates", None)
    assert parent_record == candidate_record
    assert candidate_record == split_record
    for name in ("model.safetensors", "base.safetensors", "behavior.safetensors"):
        one = load_weights(parent / name).state_dict()
        two = load_weights(candidate_native / name).state_dict()
        assert all(tensor_bits_equal(v, two[k]) for k, v in one.items())
        assert (candidate_native / name).read_bytes() == (split / name).read_bytes()
    for epoch in (1, 2):
        parent_epoch = json.loads(
            gzip.decompress((tmp_path / f"parent428/journal{epoch}.gz").read_bytes())
        )
        candidate_epoch = json.loads(
            gzip.decompress((tmp_path / f"candidate/journal{epoch}.gz").read_bytes())
        )
        assert parent_epoch["own_search"]["schema"] == "ownsearch-random-block-targets-v1"
        assert candidate_epoch["own_search"]["schema"] == "ownsearch-random-block-targets-v2"

    parent_manifest = json.loads((parent / "checkpoint.json").read_text())
    candidate_manifest = json.loads((candidate_native / "checkpoint.json").read_text())
    assert parent_manifest["source_commit"] == "8" * 40
    assert candidate_manifest["source_commit"] == SOURCE
    assert parent_manifest["run_config"] == candidate_manifest["run_config"]
    parent_state, candidate_state = parent_manifest["state"], candidate_manifest["state"]
    parent_state.pop("sample_chain_sha256", None)
    candidate_state.pop("sample_chain_sha256", None)
    assert parent_state == candidate_state

    # A source428 native is deliberately not a method7 continuation, even
    # though certificate-free model/optimizer trajectories are equivalent.
    resume_code = r"""
import json,sys,torch
from pathlib import Path
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_ownsearch_learner import TorchOwnSearchConfig,TorchOwnSearchLearner
root,cfg,weights,source=sys.argv[1:];root=Path(root);cfg=Path(cfg)
c=json.loads(cfg.read_text());c['actors']=OnlineActorConfig(**c['actors']);c['objective']=OwnSearchObjective(**c['objective']);c['search']=OwnSearchConfig(**c['search']);c['schedule']=FullGamePPOTrainConfig(**c['schedule'])
inputs=dict(initial_weights=Path(weights),book=cfg.parent/'book.json',experiment_config=cfg,protocol=cfg.parent/'protocol.json')
torch.set_num_threads(1)
try:
 TorchOwnSearchLearner.resume(root/'native2',config=TorchOwnSearchConfig(**c),input_paths=inputs,source_commit=source)
except ValueError as exc:
 assert 'source/config/runtime mismatch' in str(exc)
 print('old-native-rejected')
else:
 raise AssertionError('source428 native accepted as method7 resume')
"""
    old_sha = hashlib.sha256((parent / "checkpoint.json").read_bytes()).hexdigest()
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            resume_code,
            str(tmp_path / "parent428"),
            str(path),
            str(WEIGHTS),
            SOURCE,
        ],
        env=dict(os.environ, PYTHONPATH=str(HERE / "src"), OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == "old-native-rejected"
    assert hashlib.sha256((parent / "checkpoint.json").read_bytes()).hexdigest() == old_sha
