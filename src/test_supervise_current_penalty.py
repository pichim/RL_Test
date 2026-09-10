import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from supervise_current_penalty import supervise


class SupervisorTests(unittest.TestCase):
    def test_success_and_nonzero_exit_are_recorded(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), TemporaryDirectory() as temporary:
                directory = Path(temporary) / "monitor"
                command = [sys.executable, "-c", f"import sys; print('probe'); sys.exit({exit_code})"]
                self.assertEqual(supervise(command, directory, temporary), exit_code)
                self.assertEqual(json.loads((directory / "exited.json").read_text())["exit_code"], exit_code)
                self.assertGreater(json.loads((directory / "started.json").read_text())["child_pid"], 0)
                self.assertIn("probe", (directory / "stdout.log").read_text())
                with self.assertRaises(FileExistsError):
                    supervise(command, directory, temporary)

    def test_launch_failure_is_recorded(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "monitor"
            with self.assertRaises(OSError):
                supervise([str(Path(temporary) / "missing.exe")], directory, temporary)
            self.assertTrue((directory / "supervisor_failed.json").exists())
            self.assertFalse((directory / "exited.json").exists())


if __name__ == "__main__":
    unittest.main()
