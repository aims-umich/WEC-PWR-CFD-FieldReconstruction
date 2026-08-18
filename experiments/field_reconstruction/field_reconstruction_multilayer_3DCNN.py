"""Run the publication 3D-CNN field-reconstruction experiment.

Relative paths in a configuration file are resolved from that file's directory.
Relative paths supplied on the command line are resolved from the repository root,
so execution does not depend on the caller's working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neup_inference_of_flow.field_reconstruction import (
    DataPathsConfig,
    MaskConfig,
    Model3DConfig,
    MultiLevel3DExperimentConfig,
    SplitConfig,
    TrainingConfig,
    run_multilevel_3d_experiment,
)

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "publication_configuration.json"


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _set_override(config: dict[str, Any], expression: str) -> None:
    """Apply a dotted ``SECTION.KEY=JSON_VALUE`` override in-place."""
    try:
        dotted_key, raw_value = expression.split("=", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid --set {expression!r}; expected SECTION.KEY=VALUE") from exc
    keys = dotted_key.split(".")
    target: Any = config
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        target = target[key]
    if not isinstance(target, dict) or keys[-1] not in target:
        raise ValueError(f"Unknown configuration key: {dotted_key}")
    try:
        target[keys[-1]] = json.loads(raw_value)
    except json.JSONDecodeError:
        target[keys[-1]] = raw_value


def load_configuration(
    config_path: Path,
    input_paths: list[str] | None = None,
    overrides: list[str] | None = None,
) -> tuple[MultiLevel3DExperimentConfig, dict[str, Any]]:
    """Load, override, validate, and path-resolve an experiment configuration."""
    config_path = config_path.expanduser()
    config_path = (REPO_ROOT / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    with config_path.open(encoding="utf-8") as stream:
        document = json.load(stream)

    output = document.pop("output", {})
    for override in overrides or []:
        _set_override(document, override)
    if input_paths:
        if len(input_paths) != 4:
            raise ValueError("Exactly four --input-path arguments are required")
        document["data"]["multi_h5_paths"] = input_paths
        path_base = REPO_ROOT
    else:
        path_base = config_path.parent

    paths = document["data"].get("multi_h5_paths", [])
    if len(paths) != 4:
        raise ValueError(f"Configuration must contain exactly four HDF5 inputs; found {len(paths)}")
    document["data"]["multi_h5_paths"] = [_resolve_path(path, path_base) for path in paths]

    config = MultiLevel3DExperimentConfig(
        data=DataPathsConfig(**document["data"]),
        mask=MaskConfig(**document.get("mask", {})),
        split=SplitConfig(**document.get("split", {})),
        model=Model3DConfig(**document.get("model", {})),
        training=TrainingConfig(**document.get("training", {})),
        plot=document.get("plot", True),
    )
    return config, output


# Import-compatible effective publication defaults for supplementary drivers.
CONFIG, _PUBLICATION_OUTPUT = load_configuration(DEFAULT_CONFIG_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-level 3D field reconstruction experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="JSON configuration file.")
    parser.add_argument(
        "--input-path",
        action="append",
        help="HDF5 input (repeat exactly four times); relative paths are repository-relative.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory; relative paths are repository-relative.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help='Override a configuration value (repeatable), e.g. --set training.epochs=10.',
    )
    # Kept for compatibility with older commands and sensitivity scripts.
    parser.add_argument("--results-dir-name", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config, output = load_configuration(config_path, args.input_path, args.set)
    if args.output_dir and args.results_dir_name:
        parser.error("Use only one of --output-dir and --results-dir-name")
    if args.results_dir_name:
        output_dir = REPO_ROOT / "experiments" / "field_reconstruction" / "results" / args.results_dir_name
    else:
        output_value = args.output_dir or output.get("directory")
        if not output_value:
            parser.error("An output directory is required via --output-dir or the configuration file")
        output_dir = Path(_resolve_path(output_value, REPO_ROOT if args.output_dir else config_path.resolve().parent))

    print(json.dumps({"configuration": asdict(config), "output_dir": str(output_dir)}, indent=2))
    run_multilevel_3d_experiment(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
