import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flux3"
    / "generate.py"
)


class TestLazySafetyInitialization(unittest.TestCase):

    def test_safety_initializes_only_on_first_access(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

        models_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Models"
        )

        safety_constructor = Mock(return_value=object())

        fake_pipeline = Mock()
        fake_pipeline.text_encoder = object()

        namespace = {
            "Safety": safety_constructor,
            "Pipeline": Mock(return_value=fake_pipeline),
            "WEIGHTS": Path("weights"),
            "MODELS": {
                "distilled": {
                    "dit": "flux-3-image-klein"
                }
            },
        }

        module = ast.Module(
            body=[models_class],
            type_ignores=[]
        )
        exec(compile(module, str(SOURCE), "exec"), namespace)

        fake_flux3 = types.ModuleType("flux3")
        fake_flux3.__path__ = []

        fake_pup = types.ModuleType("flux3.pup")
        fake_pup.Upsampler = Mock()

        with patch.dict(
            sys.modules,
            {"flux3": fake_flux3, "flux3.pup": fake_pup}
        ):
            models = namespace["Models"](
                {"distilled": "dummy_checkpoint"},
                pup=None,
                pup_memory=0.35,
            )

        self.assertEqual(
            safety_constructor.call_count,
            0,
            "Safety must not initialize during Models construction.",
        )

        first = models.safety
        second = models.safety

        self.assertIs(first, second)
        self.assertEqual(safety_constructor.call_count, 1)


if __name__ == "__main__":
    unittest.main()
