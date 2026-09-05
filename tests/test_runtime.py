import subprocess
import sys
import unittest


class StageWatchdogTests(unittest.TestCase):
    def run_script(self, body):
        return subprocess.run(
            [sys.executable, "-u", "-c",
             "from src.helper.runtime import stage\nimport time\n" + body],
            capture_output=True, text=True, timeout=5,
        )

    def test_stuck_stage_exits_with_traceback(self):
        result = self.run_script(
            "with stage('FLUX', 0.1):\n    time.sleep(10)\n"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("START FLUX", result.stdout)
        self.assertIn("Timeout", result.stderr)
        self.assertNotIn("END FLUX", result.stdout)

    def test_success_and_exception_cancel_watchdog(self):
        result = self.run_script(
            "with stage('success', 0.1):\n    pass\n"
            "try:\n    with stage('failure', 0.1):\n        raise ValueError('bad')\n"
            "except ValueError:\n    pass\n"
            "time.sleep(0.3)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("END success", result.stdout)
        self.assertIn("FAILED failure", result.stdout)

    def test_nonpositive_timeout_is_rejected(self):
        result = self.run_script("with stage('invalid', 0):\n    pass\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
