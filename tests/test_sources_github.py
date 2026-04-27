"""Direct unit tests for scripts.sources.github.

Validates the GitHub-URL parser as a standalone import (not via the
``scripts.run_batch_standardization`` re-export). The orchestrator's
end-to-end use of ``_parse_github_location`` is covered by the existing
batch-discovery tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.sources.github import GITHUB_HOSTS, GitHubLocation, _parse_github_location


class GitHubLocationDataclassTests(unittest.TestCase):
    def test_default_ref_and_subpath_are_none(self):
        loc = GitHubLocation(clone_url="https://github.com/x/y.git")
        self.assertIsNone(loc.ref)
        self.assertIsNone(loc.subpath)

    def test_dataclass_is_frozen(self):
        loc = GitHubLocation(clone_url="https://github.com/x/y.git")
        with self.assertRaises(Exception):
            loc.clone_url = "evil"  # type: ignore[misc]


class ParseGithubLocationTests(unittest.TestCase):
    def test_bare_repo_url(self):
        loc = _parse_github_location("https://github.com/owner/repo")
        self.assertEqual(loc.clone_url, "https://github.com/owner/repo.git")
        self.assertIsNone(loc.ref)
        self.assertIsNone(loc.subpath)

    def test_repo_url_with_dot_git_suffix_is_stripped(self):
        loc = _parse_github_location("https://github.com/owner/repo.git")
        self.assertEqual(loc.clone_url, "https://github.com/owner/repo.git")

    def test_tree_url_with_ref(self):
        loc = _parse_github_location("https://github.com/owner/repo/tree/main")
        self.assertEqual(loc.ref, "main")
        self.assertIsNone(loc.subpath)

    def test_tree_url_with_ref_and_subpath(self):
        loc = _parse_github_location("https://github.com/owner/repo/tree/v1.2.3/data/raw")
        self.assertEqual(loc.ref, "v1.2.3")
        self.assertEqual(loc.subpath, Path("data") / "raw")

    def test_blob_url_is_accepted(self):
        loc = _parse_github_location("https://github.com/owner/repo/blob/main/file.csv")
        self.assertEqual(loc.ref, "main")
        self.assertEqual(loc.subpath, Path("file.csv"))

    def test_www_host_is_accepted(self):
        loc = _parse_github_location("https://www.github.com/owner/repo")
        self.assertEqual(loc.clone_url, "https://github.com/owner/repo.git")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_github_location("")
        self.assertIn("non-empty", str(ctx.exception))

    def test_non_github_host_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_github_location("https://gitlab.com/owner/repo")
        self.assertIn("Unsupported GitHub host", str(ctx.exception))

    def test_path_too_short_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_github_location("https://github.com/owner")
        self.assertIn("repository", str(ctx.exception))

    def test_invalid_tree_path_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_github_location("https://github.com/owner/repo/issues")
        self.assertIn("tree/blob URL", str(ctx.exception))


class GithubHostsConstantTests(unittest.TestCase):
    def test_includes_both_github_hosts(self):
        self.assertIn("github.com", GITHUB_HOSTS)
        self.assertIn("www.github.com", GITHUB_HOSTS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
