# MIHVER invariant value representation preregistration

## Mechanism

The existing value path flattens spatial features learned primarily by the policy trunk. Under the
available small-data regime it fits trajectory-specific patterns but does not generalize even a
deterministic material function. MIHVER separates value representation from policy behavior and
adds an explicit permutation-invariant route for global board information.

The experimental network retains the complete release policy/trunk/value path and adds:

- a zero-initialized residual linear projection from global means of all 104 encoded planes;
- a small policy-independent convolutional value tower over the encoded board;
- global mean and maximum pooling of that tower before a value MLP;
- a zero-initialized residual output for the tower.

Loading the release baseline must preserve policy logits and WDL logits bitwise before training.
The release policy/trunk and legacy value head remain frozen throughout the first probe.

## Frozen deterministic probe

- Same deduplicated corrected replay pool and trajectory-disjoint 148/48 split.
- Same deterministic round-robin sample: 8,192 train and 4,096 validation positions.
- Same depth-0 side-to-move material target and WDL expected-score MSE.
- Batch 64, Adam, 200 steps, validation every 20, seed `2026083061`.
- Low-dimensional residual-head learning rate: `2e-3`, fixed before results.

Arms:

1. `global-linear`: train only the 104-plane invariant residual projection.
2. `invariant-tower`: train both new residual branches; every release parameter stays frozen.

Hard gate for either arm:

- validation MSE at least 50% below release baseline;
- validation Pearson at least 0.80;
- validation MAE at most 0.05;
- release policy and legacy WDL initialization hashes/logits exact before training;
- release parameter hash exact after training.

Only a passing arm may advance. Select the simpler `global-linear` arm if both pass unless the
tower improves MSE by at least another 20% relative to the global arm.

## Subsequent gates

The selected deterministic-qualified representation will then be tested, in order, on corrected
held-out WDL/calibration, continuation action-value ranking, and existing Full Gumbel tactical
retention. Each later experiment requires its own frozen preregistration. Policy imitation or a
material-probe pass alone cannot authorize continuous learning, generation, arena, or promotion.

