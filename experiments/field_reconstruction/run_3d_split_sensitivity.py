"""Run multilevel 3D CNN split-sensitivity experiments.

This script repeats the four-axial-layer 3D reconstruction experiment from
``field_reconstruction_multilayer_3DCNN.py`` while changing only the training
fraction. Validation is held fixed at 0.10 and the remaining fraction is used
for testing. After each run, the Base Layer metrics are extracted from index 0
of ``per_level_summary.json`` and written to a compact summary file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRAIN_FRACTIONS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
VAL_FRACTION = 0.10
RESULTS_PREFIX = "split_sensitivity_3d_train"


def _results_root() -> Path:
    return REPO_ROOT / "experiments" / "field_reconstruction" / "results"


def _format_train_fraction(train_frac: float) -> str:
    """Format 0.45 as 0p45 for readable, shell-safe result folders."""
    return f"{train_frac:.2f}".replace(".", "p")


def _results_dir_name(train_frac: float) -> str:
    return f"{RESULTS_PREFIX}_{_format_train_fraction(train_frac)}"


def _config_for_train_fraction(train_frac: float):
    from field_reconstruction_multilayer_3DCNN import CONFIG

    split = replace(
        CONFIG.split,
        train_frac=train_frac,
        val_frac=VAL_FRACTION,
    )
    return replace(CONFIG, split=split)


def _load_base_layer_metrics(results_dir: Path) -> dict[str, float]:
    summary_path = results_dir / "per_level_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        per_level = json.load(f)

    return {
        "base_layer_mae": float(per_level["mae"][0]),
        "base_layer_mape": float(per_level["mape"][0]),
        "base_layer_r2": float(per_level["r2"][0]),
        "base_layer_count": float(per_level["counts"][0]),
    }


def _write_summary(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_sensitivity_3d_base_layer_summary.json"
    csv_path = output_dir / "split_sensitivity_3d_base_layer_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "results_dir_name",
        "train_frac",
        "val_frac",
        "test_frac",
        "base_layer_mae",
        "base_layer_mape",
        "base_layer_r2",
        "base_layer_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote Base Layer summary JSON: {json_path}")
    print(f"Wrote Base Layer summary CSV:  {csv_path}")


def run_split_sensitivity(train_fractions: Iterable[float], *, skip_existing: bool = False) -> list[dict[str, float | str]]:
    from neup_inference_of_flow.field_reconstruction import run_multilevel_3d_experiment

    rows: list[dict[str, float | str]] = []
    results_root = _results_root()

    for train_frac in train_fractions:
        config = _config_for_train_fraction(train_frac)
        results_dir_name = _results_dir_name(train_frac)
        results_dir = results_root / results_dir_name
        test_frac = 1.0 - train_frac - VAL_FRACTION

        if skip_existing and (results_dir / "per_level_summary.json").exists():
            print(f"Skipping existing run: {results_dir_name}")
        else:
            print(
                "Running split sensitivity experiment: "
                f"train={train_frac:.2f}, val={VAL_FRACTION:.2f}, test={test_frac:.2f} "
                f"-> {results_dir_name}"
            )
            run_multilevel_3d_experiment(config, results_dir_name=results_dir_name)

        metrics = _load_base_layer_metrics(results_dir)
        rows.append(
            {
                "results_dir_name": results_dir_name,
                "train_frac": round(train_frac, 2),
                "val_frac": VAL_FRACTION,
                "test_frac": round(test_frac, 2),
                **metrics,
            }
        )

    _write_summary(rows, results_root)
    return rows


def _parse_train_fractions(raw_values: list[str] | None) -> tuple[float, ...]:
    if not raw_values:
        return TRAIN_FRACTIONS
    return tuple(float(value) for value in raw_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multilevel 3D CNN split-sensitivity study and summarize Base Layer metrics."
    )
    parser.add_argument(
        "--train-frac",
        action="append",
        dest="train_fractions",
        help="Training fraction to run. May be supplied multiple times. Defaults to 0.45 through 0.70.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing result folders when per_level_summary.json is already present.",
    )
    args = parser.parse_args()

    run_split_sensitivity(
        _parse_train_fractions(args.train_fractions),
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
