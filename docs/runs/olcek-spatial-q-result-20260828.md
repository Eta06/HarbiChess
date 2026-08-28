# OLCEK uncertainty-weighted spatial-Q result

Date: 2026-08-28  
Corrected source commit: `438d251`  
Teacher labels: `artifacts/diagnostics/olcek-uncertainty-labels-20260828-01/labels.json`  
Transfer: `artifacts/runs/olcek-spatial-q-20260828-02/result.json`  
Decision: spatial transfer failed; completed-Q search and generation remain blocked

## Teacher qualification

The third 96/48 set excluded all TERAZI and DOKU identities. Uncertainty weighting retained continuous confidence below Q drift 0.03 and zero supervision above it.

| Validation label metric | Result | Gate |
|---|---:|---:|
| Common legal support | 97.94% | at least 95%, passed |
| Drift-qualified visit mass | 85.18% | at least 80%, passed |
| Stable Q/verifier Spearman | 0.39485 | at least 0.35, passed |
| Conservative-Q verified delta | +0.10915 | positive interval, passed |
| Delta 95% interval | [+0.05799, +0.17338] | passed |
| Harmful ratio | 4.17% | at most 10%, passed |
| Mean regret | 0.05035 | at most 0.10, passed |

The teacher gate passed completely and authorized only the frozen spatial transfer.

## Learner result

An evaluator-contract bug in the first run included zero-weight actions in a metric preregistered for supervised support. The implementation was corrected before rerunning the exact same data, seed, steps, and thresholds. Strength metrics always remained over all legal actions.

| Step | Validation Q MSE | Teacher-Q Spearman | Verified delta | Harmful | Regret | Top-16 coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.07312 | 0.000 | +0.02862 | 12.50% | 0.13089 | 54.17% |
| 60 | 0.06269 | 0.050 | +0.03024 | 12.50% | 0.12927 | 77.08% |
| 120 | 0.06062 | 0.002 | +0.00020 | 12.50% | 0.15931 | 79.17% |
| 240 | 0.05900 | 0.007 | +0.01118 | 14.58% | 0.14833 | 83.33% |
| 480 | 0.05829 | -0.016 | -0.00129 | 20.83% | 0.16079 | 83.33% |

Step 480 narrowly achieved the required 20% MSE reduction, proving that the compact head can fit average Q magnitude. It failed action ranking, independent strength, harmful-action, and regret gates. Policy/WDL logits remained bitwise identical and gradients were finite.

## Representation diagnosis

The spatial `1×1` head shares parameters but computes each action advantage only from the canonical origin-square feature. It cannot directly condition on the destination square. MSE can improve by learning move-plane and state-average biases without learning which destination is tactically or positionally good, exactly matching the observed ranking failure.

The next representation should combine origin trunk features, destination trunk features, and a move-plane embedding through a small shared scorer. This preserves spatial parameter sharing and low capacity while expressing the actual state-action relation. It requires another fully fresh label set; OLCEK duration, cutoff, and head are rejected rather than retuned.
