# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Check the submission contract without a running Slurm cluster."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class WorkflowChecks(unittest.TestCase):
    def submit(self, directory, fail_at=0):
        """Capture scheduler requests and optionally reject a submission.

Args:
    directory (str): Temporary workspace for the fake scheduler.
    fail_at (int, optional): Submission number to reject, or zero for success.

Returns:
    result (CompletedProcess): Workflow command result.
    requests (list): Arguments received by the scheduler.
    output (Path): Workflow output directory.
"""
        root = Path(directory)
        log = root / "requests.json"
        log.write_text("[]")
        scheduler = root / "sbatch"
        scheduler.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
log = Path(os.environ["SUBMISSION_LOG"])
requests = json.loads(log.read_text())
requests.append(sys.argv[1:])
log.write_text(json.dumps(requests))
if len(requests) == int(os.environ["FAIL_AT"]):
    sys.exit("submission rejected")
print(100 + len(requests))
''')
        scheduler.chmod(0o755)
        program = root / "input program.json"
        program.write_text('{"example": "submission fixture"}\n')
        output = root / "output with spaces"
        env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                   SUBMISSION_LOG=str(log), FAIL_AT=str(fail_at),
                   GPU_REQUEST="--gres=gpu:1 --partition=gpu",
                   QPU_REQUEST="--resources=PASQAL_LOCAL:1 --nodelist=c2",
                   ANALYSIS_REQUEST="--account=training --partition=cpu")
        script = Path(__file__).resolve().parents[1] / "04-slurm/gpu_to_qpu.sh"
        result = subprocess.run(["bash", str(script), str(program), str(output)],
                                env=env, capture_output=True, text=True)
        self.assertEqual((output / "program.json").read_bytes(), program.read_bytes())
        return result, json.loads(log.read_text()), output

    def test_resource_separation_and_dependencies(self):
        """Reserve GPU and QPU separately and run analysis only after successful predecessors."""
        with tempfile.TemporaryDirectory() as directory:
            result, requests, output = self.submit(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(requests), 3)
            gpu, qpu, analysis = requests
            self.assertIn("--gres=gpu:1", gpu)
            self.assertIn("--emulate", gpu)
            self.assertFalse(any("resources=" in option for option in gpu))
            self.assertIn("--resources=PASQAL_LOCAL:1", qpu)
            self.assertNotIn("--emulate", qpu)
            self.assertFalse(any("gres=" in option for option in qpu))
            self.assertIn("--dependency=afterok:101", qpu)
            self.assertIn("--dependency=afterok:102", analysis)
            for request in (qpu, analysis):
                self.assertIn("--kill-on-invalid-dep=yes", request)
            self.assertIn("--account=training", analysis)
            self.assertIn("--partition=cpu", analysis)
            self.assertFalse(any("gres=" in option or "resources=" in option for option in analysis))
            self.assertIn(str(output / "program.json"), gpu)
            self.assertEqual((output / "job-ids.txt").read_text().splitlines(), ["101", "102", "103"])

    def test_partial_submission_keeps_job_ids(self):
        """Keep accepted job IDs available when a later submission is rejected."""
        for fail_at in (2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                result, requests, output = self.submit(directory, fail_at)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(requests), fail_at)
                self.assertEqual((output / "job-ids.txt").read_text().splitlines(),
                                 [str(100 + n) for n in range(1, fail_at)])


if __name__ == "__main__":
    unittest.main()
