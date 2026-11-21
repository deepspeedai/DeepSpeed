import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_cuh_ignore.py"
_SPEC = importlib.util.spec_from_file_location("check_cuh_ignore", SCRIPT_PATH)
check_cuh_ignore = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_cuh_ignore)


class TestCheckCuhIgnore(unittest.TestCase):

    def _write_required_header(self, repo_root: Path):
        required_header = repo_root / "csrc" / "adam" / "multi_tensor_apply.cuh"
        required_header.parent.mkdir(parents=True, exist_ok=True)
        required_header.write_text("// test header\n", encoding="utf-8")

    def test_validate_repo_rejects_global_cuh_ignore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".gitignore").write_text("*.cuh\n", encoding="utf-8")
            self._write_required_header(repo_root)

            with mock.patch.object(check_cuh_ignore, "_is_git_tracked", return_value=True):
                errors = check_cuh_ignore.validate_repo(repo_root)

        self.assertTrue(any("Forbidden .gitignore pattern '*.cuh'" in error for error in errors))

    def test_validate_repo_accepts_tracked_required_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / ".gitignore").write_text("*.hip\n", encoding="utf-8")
            self._write_required_header(repo_root)

            with mock.patch.object(check_cuh_ignore, "_is_git_tracked", return_value=True):
                errors = check_cuh_ignore.validate_repo(repo_root)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
