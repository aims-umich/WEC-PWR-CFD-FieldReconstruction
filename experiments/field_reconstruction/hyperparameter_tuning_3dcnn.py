import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neup_inference_of_flow.field_reconstruction import (
    DataPathsConfig,
    MaskConfig,
    Model3DConfig,
    MultiLevel3DExperimentConfig,
    SplitConfig,
    TrainingConfig,
    run_multilevel_3d_experiment,
)


def build_base_config():
    return MultiLevel3DExperimentConfig(
        data=DataPathsConfig(
            multi_h5_paths=(
                "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_01_Base.h5",
                "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_02_Layer1.h5",
                "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_03_Layer2.h5",
                "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_04_Layer3.h5",
            ),
        ),
        mask=MaskConfig(frac_missing=0.50, pattern="checkerboard", stride=None, axis="row", seed=42),
        split=SplitConfig(train_frac=0.30, val_frac=0.10, seed=123),
        model=Model3DConfig(base_channels=96, dilations_xy=(1, 2, 3, 3, 2, 1), kdepth=3, add_coords_3d=True),
        training=TrainingConfig(
            epochs=50,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=None,
            use_amp=False,
            patience=5,
            lr_factor=0.5,
            seed=1337,
            batch_size=64,
            num_workers=0,
        ),
        plot=True,
    )


def _timestamp_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _objective_from_result(result):
    history = result["history"]
    per_level = result["per_level"]
    min_val_loss = min(history["val"]) if history.get("val") else float("inf")
    mean_level_mae = float(np.nanmean(np.asarray(per_level["mae"], dtype=np.float64)))
    return min_val_loss, mean_level_mae


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _append_csv(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _trial_overrides_for_phase(phase, rng):
    if phase == "phase_a":
        return {
            "training.lr": 10 ** rng.uniform(-4, np.log10(3e-3)),
            "training.weight_decay": 10 ** rng.uniform(-6, np.log10(3e-3)),
            "training.batch_size": rng.choice([32, 64, 96, 128]),
            "model.base_channels": rng.choice([48, 64, 96, 128]),
        }
    if phase == "phase_b":
        return {
            "model.kdepth": rng.choice([3, 5, 7]),
            "model.dilations_xy": rng.choice([
                (1, 2, 3, 3, 2, 1),
                (1, 2, 4, 4, 2, 1),
                (1, 2, 3, 4, 3, 2),
                (1, 1, 2, 3, 2, 1),
            ]),
            "model.add_coords_3d": rng.choice([True, False]),
            "model.base_channels": rng.choice([64, 96, 128]),
        }
    if phase == "phase_c":
        return {
            "training.patience": rng.choice([3, 5, 8]),
            "training.lr_factor": rng.choice([0.2, 0.5, 0.8]),
            "training.epochs": rng.choice([40, 50, 60]),
        }
    raise ValueError(f"Unknown phase: {phase}")


def _apply_overrides(config, overrides):
    model_kwargs = {}
    training_kwargs = {}
    split_kwargs = {}
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        if section == "model":
            model_kwargs[name] = value
        elif section == "training":
            training_kwargs[name] = value
        elif section == "split":
            split_kwargs[name] = value
        else:
            raise ValueError(f"Unsupported override section: {section}")

    model = replace(config.model, **model_kwargs) if model_kwargs else config.model
    training = replace(config.training, **training_kwargs) if training_kwargs else config.training
    split = replace(config.split, **split_kwargs) if split_kwargs else config.split
    return replace(config, model=model, training=training, split=split)


def _configure_logger(log_path):
    logger = logging.getLogger("tuning_3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run_tuning(run_name, phase_trials, seed):
    root_dir = Path("experiments") / "field_reconstruction" / "results" / run_name
    root_dir.mkdir(parents=True, exist_ok=True)

    logger = _configure_logger(root_dir / "tuning.log")
    rng = random.Random(seed)

    phases = [
        ("phase_a", phase_trials[0]),
        ("phase_b", phase_trials[1]),
        ("phase_c", phase_trials[2]),
    ]

    manifest_path = root_dir / "tuning_manifest.json"
    jsonl_path = root_dir / "trial_results.jsonl"
    csv_path = root_dir / "trial_results.csv"
    best_path = root_dir / "best_so_far.json"

    base_config = build_base_config()
    best_config = base_config
    best_score = float("inf")
    best_trial_id = None

    _write_json(
        manifest_path,
        {
            "created_utc": _timestamp_utc(),
            "run_name": run_name,
            "seed": seed,
            "phase_trials": {"phase_a": phase_trials[0], "phase_b": phase_trials[1], "phase_c": phase_trials[2]},
            "base_config": asdict(base_config),
        },
    )

    fieldnames = [
        "trial_id",
        "phase",
        "status",
        "min_val_loss",
        "mean_level_mae",
        "results_dir",
        "error",
    ]

    logger.info("Starting 3D CNN hyperparameter tuning run '%s'.", run_name)

    trial_counter = 0
    for phase_name, n_trials in phases:
        logger.info("Starting %s with %d trials.", phase_name, n_trials)
        for i in range(1, n_trials + 1):
            trial_counter += 1
            trial_id = f"{phase_name}_trial_{i:03d}"
            overrides = _trial_overrides_for_phase(phase_name, rng)
            trial_config = _apply_overrides(best_config, overrides)
            trial_results_dir_name = f"{run_name}/{phase_name}/{trial_id}"

            logger.info("[%s] Starting trial with overrides: %s", trial_id, overrides)
            trial_record = {
                "timestamp_utc": _timestamp_utc(),
                "trial_id": trial_id,
                "phase": phase_name,
                "trial_index": trial_counter,
                "overrides": overrides,
                "status": "running",
            }
            _append_jsonl(jsonl_path, trial_record)

            try:
                result = run_multilevel_3d_experiment(trial_config, results_dir_name=trial_results_dir_name)
                min_val_loss, mean_level_mae = _objective_from_result(result)

                summary = {
                    "timestamp_utc": _timestamp_utc(),
                    "trial_id": trial_id,
                    "phase": phase_name,
                    "status": "ok",
                    "min_val_loss": min_val_loss,
                    "mean_level_mae": mean_level_mae,
                    "overrides": overrides,
                    "results_dir": result.get("results_dir", ""),
                }
                _append_jsonl(jsonl_path, summary)
                _append_csv(
                    csv_path,
                    {
                        "trial_id": trial_id,
                        "phase": phase_name,
                        "status": "ok",
                        "min_val_loss": min_val_loss,
                        "mean_level_mae": mean_level_mae,
                        "results_dir": result.get("results_dir", ""),
                        "error": "",
                    },
                    fieldnames=fieldnames,
                )

                logger.info(
                    "[%s] Completed: min_val_loss=%.6e | mean_level_mae=%.6e",
                    trial_id,
                    min_val_loss,
                    mean_level_mae,
                )

                if min_val_loss < best_score:
                    best_score = min_val_loss
                    best_config = trial_config
                    best_trial_id = trial_id
                    _write_json(
                        best_path,
                        {
                            "updated_utc": _timestamp_utc(),
                            "best_trial_id": best_trial_id,
                            "best_min_val_loss": _safe_float(best_score),
                            "best_config": asdict(best_config),
                        },
                    )
                    logger.info("[%s] New best trial with min_val_loss=%.6e", trial_id, best_score)
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)
                logger.exception("[%s] Trial failed with error: %s", trial_id, err_msg)
                _append_jsonl(
                    jsonl_path,
                    {
                        "timestamp_utc": _timestamp_utc(),
                        "trial_id": trial_id,
                        "phase": phase_name,
                        "status": "failed",
                        "error": err_msg,
                        "overrides": overrides,
                    },
                )
                _append_csv(
                    csv_path,
                    {
                        "trial_id": trial_id,
                        "phase": phase_name,
                        "status": "failed",
                        "min_val_loss": "",
                        "mean_level_mae": "",
                        "results_dir": "",
                        "error": err_msg,
                    },
                    fieldnames=fieldnames,
                )

        logger.info("Completed %s.", phase_name)

    _write_json(
        root_dir / "tuning_complete.json",
        {
            "completed_utc": _timestamp_utc(),
            "run_name": run_name,
            "best_trial_id": best_trial_id,
            "best_min_val_loss": _safe_float(best_score),
            "best_config": asdict(best_config),
        },
    )
    logger.info("Tuning complete. Best trial: %s (min_val_loss=%.6e)", best_trial_id, best_score)


def main():
    parser = argparse.ArgumentParser(description="Three-phase hyperparameter tuning for 3D CNN field reconstruction.")
    parser.add_argument("--run-name", required=True, help="Results root folder name under experiments/field_reconstruction/results/.")
    parser.add_argument("--phase-a-trials", type=int, default=20, help="Number of trials for phase A.")
    parser.add_argument("--phase-b-trials", type=int, default=15, help="Number of trials for phase B.")
    parser.add_argument("--phase-c-trials", type=int, default=10, help="Number of trials for phase C.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for trial sampling.")
    args = parser.parse_args()

    run_tuning(args.run_name, (args.phase_a_trials, args.phase_b_trials, args.phase_c_trials), args.seed)


if __name__ == "__main__":
    main()
