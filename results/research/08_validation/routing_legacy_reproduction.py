"""Expected failing fixture for the preserved legacy routing summary.

Execute only the original scalar expression on a minimal tensor-like fixture;
no Torch, model, task examples, random draws or training are involved.
"""
import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("routing_phase3_original.py")
TREE = ast.parse(SOURCE.read_text())
GROW2 = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "t_grow2")
EXPRESSION = next(node.value for node in ast.walk(GROW2) if isinstance(node, ast.Assign)
                  and ast.unparse(node.value) == "pick.float().mean().item()")


class FixturePick:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def mean(self):
        return FixturePick(sum(self.values) / len(self.values))

    def item(self):
        return self.values


def legacy_summary(values):
    return eval(compile(ast.Expression(EXPRESSION), str(SOURCE), "eval"),
                {"__builtins__": {}}, {"pick": FixturePick(values)})


class LegacyRoutingFixture(unittest.TestCase):
    def test_fixture_executes_the_preserved_source_expression(self):
        self.assertEqual(ast.unparse(EXPRESSION), "pick.float().mean().item()")
        self.assertEqual(legacy_summary([1, 1]), 1.0)

    def test_distinct_routing_distributions_are_distinguishable(self):
        # All slot 1 and an equal mixture of slots 0/2 are different assignments.
        self.assertNotEqual(legacy_summary([1, 1]), legacy_summary([0, 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
