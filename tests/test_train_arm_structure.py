import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrainArmStructureTests(unittest.TestCase):
    def test_no_classes_inside_functions(self):
        # Trainer.save_model pickles the config; pickle cannot find local classes.
        tree = ast.parse((ROOT / "scripts/train_arm.py").read_text())
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = [n.name for n in ast.walk(func) if isinstance(n, ast.ClassDef)]
                self.assertEqual(local, [], f"Local classes in {func.name}: {local}")

    def test_helpers_are_module_level(self):
        tree = ast.parse((ROOT / "src/grpo_compat.py").read_text())
        names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        self.assertEqual(names, {"CompatibleGRPOConfig", "TrainingTimeBudget"})


if __name__ == "__main__":
    unittest.main()
