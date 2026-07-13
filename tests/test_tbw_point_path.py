"""Path portability regression for the frozen-weight TBW point runner."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "mechanism_influence" / "tbw_point.py"


class TBWPointPathTests(unittest.TestCase):
    def test_bundle_is_repository_root_relative_to_script_file(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(SOURCE))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "BUNDLE" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "relocated-repository"
            script = repository / "mechanism_influence" / "tbw_point.py"
            expression = ast.Expression(assignments[0].value)
            bundle = eval(
                compile(expression, str(SOURCE), "eval"),
                {"os": os, "__file__": str(script)},
            )
            self.assertEqual(Path(bundle), repository)
        self.assertNotIn("/home/vishnu/coding_proj/fsts_5", source)


if __name__ == "__main__":
    unittest.main()
