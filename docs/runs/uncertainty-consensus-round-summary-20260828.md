# Uncertainty-consensus development round summary (2026-08-28)

## Outcome

The 512/800 conservative-Q, KL-constrained SIPER target is independently
qualified, but no learner checkpoint is qualified. Learner, arena, generation,
and promotion remain blocked. The next decision is whether to preregister a
train-only model-update trust-region/safety projection before one fresh
validation test.

## Commit-to-file manifest

- `b93b0f2` — `docs/runs/deger-action-value-preregistration-20260828.md`
- `3ad6ef5` — `src/harbichess/backends/action_value_network.py`
- `6f9f314` — `tests/test_action_value_network.py`
- `96eb79a` — `src/harbichess/training/action_value_transfer.py`
- `f83418b` — `tests/test_action_value_transfer.py`
- `3fffa8f` — `docs/runs/deger-action-value-result-20260828.md`
- `abaf60c` — `docs/runs/doku-spatial-q-preregistration-20260828.md`
- `4579a46` — `src/harbichess/evaluation/action_value_dataset.py`
- `bf56220` — `tests/test_action_value_dataset.py`
- `551ad77` — `docs/runs/doku-spatial-q-result-20260828.md`
- `6548c23` — `docs/runs/olcek-uncertainty-q-preregistration-20260828.md`
- `ed9cc9c` — `src/harbichess/backends/action_value_network.py`
- `7320e0f` — `tests/test_action_value_network.py`
- `3d0ab0c` — `src/harbichess/evaluation/action_value_dataset.py`
- `e374e25` — `tests/test_action_value_dataset.py`
- `d879ace` — `src/harbichess/evaluation/uncertainty_q_labels.py`
- `e160d1f` — `tests/test_uncertainty_q_labels.py`
- `a27e478` — `src/harbichess/training/spatial_action_value_transfer.py`
- `502c2ec` — `tests/test_spatial_action_value_transfer.py`
- `dc25a81` — `src/harbichess/training/spatial_action_value_transfer.py`
- `438d251` — `tests/test_spatial_action_value_transfer.py`
- `e49f944` — `docs/runs/olcek-spatial-q-result-20260828.md`
- `ae7e6c1` — `docs/runs/bag-move-conditioned-q-preregistration-20260828.md`
- `e0c5b70` — `src/harbichess/chess/actions.py`
- `203507a` — `tests/test_actions.py`
- `11832c8` — `src/harbichess/backends/action_value_network.py`
- `2427983` — `tests/test_action_value_network.py`
- `244c087` — `src/harbichess/evaluation/action_value_dataset.py`
- `951034d` — `src/harbichess/evaluation/uncertainty_q_labels.py`
- `f163f02` — `src/harbichess/training/spatial_action_value_transfer.py`
- `039237b` — `tests/test_spatial_action_value_transfer.py`
- `4fc3ad7` — `docs/runs/bag-move-conditioned-q-result-20260828.md`
- `9505532` — `docs/runs/akis-uncertainty-policy-preregistration-20260828.md`
- `03befea` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `edcdffb` — `tests/test_uncertainty_policy_transfer.py`
- `18ba761` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `961966c` — `docs/runs/akis-uncertainty-policy-result-20260828.md`
- `5e0742e` — `docs/runs/veri-coverage-transfer-preregistration-20260828.md`
- `bb4f32c` — `src/harbichess/evaluation/action_value_dataset.py`
- `1b59ceb` — `src/harbichess/evaluation/uncertainty_q_labels.py`
- `a6f8a8c` — `tests/test_uncertainty_q_labels.py`
- `8af17c9` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `b693770` — `docs/runs/veri-coverage-transfer-result-20260828.md`
- `d300243` — `docs/runs/kilavuz-policy-improvement-preregistration-20260828.md`
- `6562067` — `src/harbichess/evaluation/policy_improvement_target.py`
- `a9d2368` — `tests/test_policy_improvement_target.py`
- `1bfa956` — `docs/runs/kilavuz-policy-improvement-result-20260828.md`
- `8c3899d` — `docs/runs/siper-conservative-policy-preregistration-20260828.md`
- `476e88d` — `src/harbichess/evaluation/policy_improvement_target.py`
- `caab547` — `tests/test_policy_improvement_target.py`
- `f22b5f3` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `6ebf386` — `tests/test_uncertainty_policy_transfer.py`
- `dbecdfe` — `docs/runs/siper-conservative-policy-result-20260828.md`
- `d8a9bcc` — `docs/runs/aktarim-fit-preregistration-20260828.md`
- `047316f` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `0b7e5bb` — `src/harbichess/training/uncertainty_policy_transfer.py`
- `06fa099` — `docs/runs/aktarim-fit-result-20260828.md`

## Verification

- `uv run ruff check .` — passed
- `uv run pytest -q` — 252 passed in 1.87 seconds
- `git status --branch --short` — `main...origin/main`, clean
- Every commit above changes one file and was pushed to `origin/main`.
