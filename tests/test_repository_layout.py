import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_setuptools_discovers_only_project_package_from_src(self):
        with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
            configuration = tomllib.load(stream)

        discovery = configuration["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(discovery["where"], ["src"])
        self.assertEqual(discovery["include"], ["neup_inference_of_flow*"])

    def test_experiment_directories_exist(self):
        expected = (
            "field_reconstruction",
            "one_step_prediction/lstm",
            "one_step_prediction/convlstm",
            "one_step_prediction/deeponet",
        )
        for relative_path in expected:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((REPO_ROOT / "experiments" / relative_path).is_dir())

    def test_publication_configuration_is_experiment_owned(self):
        experiment_config = REPO_ROOT / "experiments/field_reconstruction/publication_configuration.json"
        package_root = REPO_ROOT / "src/neup_inference_of_flow"

        self.assertTrue(experiment_config.is_file())
        self.assertFalse(any(package_root.rglob("publication_configuration.json")))

    def test_reusable_field_reconstruction_modules_stay_in_src(self):
        package_root = REPO_ROOT / "src/neup_inference_of_flow/field_reconstruction"
        experiment_root = REPO_ROOT / "experiments/field_reconstruction"

        for module_name in ("datasets.py", "metrics.py", "pipelines.py", "training.py"):
            with self.subTest(module_name=module_name):
                self.assertTrue((package_root / module_name).is_file())
                self.assertFalse((experiment_root / module_name).exists())


if __name__ == "__main__":
    unittest.main()
