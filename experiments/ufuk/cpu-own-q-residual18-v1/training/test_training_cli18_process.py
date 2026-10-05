import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from residual_critic18 import ResidualCritic18
from train_residual18 import ResidualTrainer, native_digest


class FreshProcessResumeTests(unittest.TestCase):
    def test_fresh_interpreter_resume_matches_whole_training(self):
        groups = {}
        for game in range(16):
            groups[f"trajectory-{game:02d}"] = [
                {
                    "x": np.eye(18, dtype=np.float64)[(game + row) % 18],
                    "prior_logit": (game - 8) / 100,
                    "target": -0.4 if row % 2 else 0.6,
                }
                for row in range(64)
            ]
        contract = {
            "updates": 64,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "clip_norm": 5.0,
            "anchor_weight": 0.1,
            "weight_l2": 1e-4,
            "prior_sha256": "a" * 64,
            "labels_sha256": "b" * 64,
            "training_dataset_sha256": "c" * 64,
            "feature_helper_sha256": "d" * 64,
            "source_commit": "e" * 40,
            "input_count": 18,
            "hidden_count": 32,
        }
        whole = ResidualCritic18(seed=20262705, contract=contract)
        ResidualTrainer(whole, groups).advance(8)

        paused = ResidualCritic18(seed=20262705, contract=contract)
        ResidualTrainer(paused, groups).advance(4)
        native = paused.native()
        wire_groups = {
            game: [
                {
                    "x": row["x"].tolist(),
                    "prior_logit": row["prior_logit"],
                    "target": row["target"],
                }
                for row in rows
            ]
            for game, rows in groups.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "pause.json"
            groups_path = root / "groups.json"
            output_path = root / "resume.json"
            state_path.write_text(json.dumps(native, separators=(",", ":")))
            groups_path.write_text(json.dumps(wire_groups, separators=(",", ":")))
            code = (
                "import json,sys; from pathlib import Path; "
                "from training_cli18_v2 import resume_native; "
                "from train_residual18 import native_digest; "
                "state=json.loads(Path(sys.argv[1]).read_text()); "
                "groups=json.loads(Path(sys.argv[2]).read_text()); "
                "contract=json.loads(Path(sys.argv[3]).read_text()); "
                "model=resume_native(state,groups,seed=20262705,contract=contract); "
                "Path(sys.argv[4]).write_text(json.dumps(model,separators=(',',':'))); "
                "print(native_digest(model))"
            )
            contract_path = root / "contract.json"
            contract_path.write_text(json.dumps(contract, separators=(",", ":")))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(state_path),
                    str(groups_path),
                    str(contract_path),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            resumed = json.loads(output_path.read_text())
        self.assertEqual(completed.stdout.strip(), native_digest(whole.native()))
        self.assertEqual(native_digest(resumed), native_digest(whole.native()))


if __name__ == "__main__":
    unittest.main()
