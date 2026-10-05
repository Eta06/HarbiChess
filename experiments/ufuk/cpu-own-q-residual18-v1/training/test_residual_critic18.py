import math
import unittest

import numpy as np

from residual_critic18 import INPUTS, ResidualCritic18


class ResidualCritic18Tests(unittest.TestCase):
    def test_zero_output_is_exact_prior_and_update_is_resumable(self):
        contract = {
            "prior_sha256": "a" * 64,
            "features": "18scaled-only-v1",
            "updates": 64,
        }
        model = ResidualCritic18(seed=7, contract=contract)
        x = np.zeros((5, INPUTS))
        base = np.array([-0.7, -0.1, 0.25, 0.9, -0.0])
        prediction, residual = model.predict(x, base)
        np.testing.assert_array_equal(prediction, np.asarray([math.tanh(value) for value in base]))
        self.assertEqual(np.signbit(prediction[-1]), np.signbit(base[-1]))
        np.testing.assert_array_equal(residual, np.zeros(len(base)))
        x[:, 0] = 1.0
        result = model.train_batch(x, base, np.array([1.0, -1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(result["step"], 1)
        restored = ResidualCritic18(seed=99, contract=contract, state=model.native())
        for key in model.params:
            np.testing.assert_array_equal(restored.params[key], model.params[key])
            np.testing.assert_array_equal(restored.m[key], model.m[key])
            np.testing.assert_array_equal(restored.v[key], model.v[key])
        np.testing.assert_array_equal(restored.predict(x, base)[0], model.predict(x, base)[0])

    def test_rejects_contract_drift_and_unclipped_targets(self):
        model = ResidualCritic18(seed=1, contract={"prior_sha256": "b" * 64, "updates": 64})
        native = model.native()
        with self.assertRaises(ValueError):
            ResidualCritic18(seed=1, contract={"prior_sha256": "c" * 64}, state=native)
        with self.assertRaises(ValueError):
            model.train_batch(
                np.zeros((1, INPUTS)), np.zeros(1), np.array([2.0])
            )


if __name__ == "__main__":
    unittest.main()
