# Fresh actor controller proposal — no actual NN actor run performed

`qualify_actor.py` is root-only actualCPU qualification, with no synthetic command flag. It requires real pinned e8 snapshot and e8 anchor, clean source6fcc8b476d25495d1c9c413e55b2c7ba4794013e, source/model/helper/config SHA closure, one CPU and one fixed observed-first+600 deadline below2026-10-06T08:00UTC. It sequentially launches whole9, pause4, then a genuinely fresh process resume9. All three journals receive independent legal-history/RNG/anchor-structure replay. Whole/resume compressed bytes and full state (including actor RNG and anchors) must match exactly. A separate full masked e8 network forward checks every stored anchor row across all three journals (22 row checks, at most9 distinct complete histories, cached only by exact full history). The receipt makes no old learner/optimizer resume or strength claim. Actual newly generated actions are9+4+5=18; no search trajectory is mislabeled as old work.

`run_epoch.py` accepts only that actual qualified controller/support/helper/model closure, re-verifies all three qualification artifact SHAs and matching fixed settings. Its new prospective config has fixed max8192 and observed-first+14400, fresh registered seed/epoch ID and fixed frozen e8. It sequentially collects cumulative1024/2048/.../8192 milestones. Each producer runs in a fresh process under the same original deadline and immutable config; parent checkpoint path+SHA is attached to the command/milestone receipt. No automatic training, epoch/model cycling, candidate choice, early outcome selection or budget reset occurs. Failure preserves closed checkpoints/logs and does not retry. There is no automatic controller restart protocol: root may explicitly design a later version if needed, preserving original clock and artifacts.

Helper inventory **must** retain the historical adjacent journal_v1.py, alongside journal_v2.py, produce_v2.py, anchor_value.py, search.py and value.py. Actual producer sibling paths for the current journal_v2 and preserved journal_v1 files are verified, not just names. Current MAIN journal_v2 is self-contained; preserved v1 is provenance, not a claimed live import. The old source-owned `experiments/ufuk/cpu-fix-v1/qualify_cli.py` supplies clean-source and publish-once utilities; sparse checkout must retain that file. Producer/library inference source stays separate from supplemental controller commits.

Artifact journals live under a new immutable `/dev/shm/...` stage capped16MiB; workspace keeps small contract/command/result/milestone receipts/logs. The guard checks workspace free>=256MiB and source6fcc `CgroupMemoryBudget(15GiB)`, whose inactive-file-v1 estimate retains active cache, physical charge and OOM protections. Both parent and owned child inherit one CPU affinity and thread limits. Only a child started by this controller is signalled on deadline/resource/error, with PID/startticks, real returncode and failure snapshot receipts. No process-name kill or unrelated owner signal exists.

Both scripts default to preflight and need explicit `--execute`. Root samples the actual first **before** config/manifest publication and includes preparation time in the shared deadline. Manifest format:

```json
{
  "schema": "fresh-qsearch-realCPU-controller-manifest-v1",
  "source_commit": "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
  "checkout": "/workspace/work/harbichess/cpu-additive-source-6fcc8b4",
  "cpu_core": "ROOT_INTEGER_ALLOWED_CORE_DISTINCT_PER_CONCURRENT_ACTOR",
  "config": {"path": "ROOT_FROZEN_CONFIG", "sha256": "ROOT_ACTUAL_SHA"},
  "model": {"path": "ROOT_REAL_E8", "sha256": "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"},
  "anchor_model": {"path": "ROOT_SAME_E8", "sha256": "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"},
  "helpers": {
    "journal_v1.py": {"path": "SIBLING_JOURNAL_V1", "sha256": "ACTUAL"},
    "journal_v2.py": {"path": "SIBLING_JOURNAL_V2", "sha256": "ACTUAL"},
    "produce_v2.py": {"path": "PINNED_PRODUCER", "sha256": "ACTUAL"},
    "anchor_value.py": {"path": "PINNED_ANCHOR", "sha256": "ACTUAL"},
    "search.py": {"path": "PINNED_SEARCH", "sha256": "ACTUAL"},
    "value.py": {"path": "PINNED_VALUE", "sha256": "ACTUAL"}
  }
}
```

Root fills actual config fields expected by MAIN journal_v2/produce_v2; qualification max9/deadline600, production max8192/deadline14400. Model/evaluator/search/anchor/runtime fields stay identical. CPU qualification is real hardware evidence only when this controller actually completes, never because pure tests pass. MAIN corrected `convert_verified` returns six outputs: features, labels, anchors, groups, provenance, receipt. This controller does not call the historical v1 converter or discard anchors.

Root command templates (not executed here):

```sh
PYTHON qualify_actor.py --manifest QUAL_MANIFEST --manifest-sha256 QUAL_SHA \
  --first-epoch OBSERVED_QUAL_FIRST --ram-stage /dev/shm/NEW_QUAL_STAGE \
  --output NEW_QUAL_RECEIPTS --execute
PYTHON run_epoch.py --manifest EPOCH_MANIFEST --manifest-sha256 EPOCH_SHA \
  --qualification ACTUAL_QUAL_RESULT --qualification-sha256 ACTUAL_QUAL_SHA \
  --first-epoch ORIGINAL_EPOCH_FIRST --ram-stage /dev/shm/NEW_EPOCH_STAGE \
  --output NEW_EPOCH_RECEIPTS --execute
```

Tests use tiny fake command/JSON/parent files only. They check actual producer flag names, absence of synthetic mode, exact parent SHA, original clock/futurefirst/operator ceiling, fixed target bound and changed-config rejection. No actor, model, game, training, network or GPU is launched by tests.
