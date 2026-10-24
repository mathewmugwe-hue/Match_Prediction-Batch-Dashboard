"""Lightweight tests for the batch fixture parser without importing Streamlit or the ML stack."""
import ast
import csv
import re
import unittest
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[1] / "dashboard.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
wanted = {"parse_neutral", "parse_fixture"}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
namespace = {"csv": csv, "re": re, "Any": Any}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
parse_fixture = namespace["parse_fixture"]


class FixtureParserTests(unittest.TestCase):
    def test_simple_fixture(self):
        row = parse_fixture("Argentina vs Brazil", 1)
        self.assertEqual((row["Home team"], row["Away team"]), ("Argentina", "Brazil"))
        self.assertIsNone(row["Home odds"])

    def test_fixture_with_odds(self):
        row = parse_fixture("Argentina vs Brazil, 2.10, 3.20, 3.60", 1)
        self.assertEqual((row["Home team"], row["Away team"]), ("Argentina", "Brazil"))
        self.assertEqual((row["Home odds"], row["Draw odds"], row["Away odds"]), (2.1, 3.2, 3.6))

    def test_tournament_with_odds(self):
        row = parse_fixture("FIFA World Cup, Argentina, Brazil, 2.10, 3.20, 3.60", 1)
        self.assertEqual(row["Tournament"], "FIFA World Cup")
        self.assertEqual((row["Home team"], row["Away team"]), ("Argentina", "Brazil"))
        self.assertEqual(row["Away odds"], 3.6)

    def test_date_tournament_teams_and_odds(self):
        row = parse_fixture("2026-11-12, FIFA World Cup, Argentina, Brazil, 2.10, 3.20, 3.60", 1)
        self.assertEqual(row["Fixture date"], "2026-11-12")
        self.assertEqual(row["Home odds"], 2.1)

    def test_neutral_venue_without_odds(self):
        row = parse_fixture("2026-11-12, FIFA World Cup, Argentina, Brazil, neutral", 1)
        self.assertTrue(row["Neutral venue"])
        self.assertIsNone(row["Home odds"])

    def test_pipe_delimited_odds(self):
        row = parse_fixture("Argentina vs Brazil | 2.10 | 3.20 | 3.60", 1)
        self.assertEqual(row["Home odds"], 2.1)
        self.assertEqual(row["Away team"], "Brazil")

    def test_malformed_input_fails_clearly(self):
        with self.assertRaises(ValueError):
            parse_fixture("no valid fixture", 1)


if __name__ == "__main__":
    unittest.main()
