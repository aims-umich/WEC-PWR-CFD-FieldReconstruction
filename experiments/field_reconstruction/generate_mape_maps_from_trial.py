import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from neup_inference_of_flow.field_reconstruction import (
    DEFAULT_GEOMETRY_MASK,
    Core3DInpaintNet,
    build_datasets_and_loaders,
    build_observed_and_missing_masks,
    compute_per_level_percentage_maps,
    fit_transform_per_level_observed_data,
    load_multiple_levels,
    make_time_splits,
    plot_per_level_maps,
)


def _configure_logger():
    logger = logging.getLogger("mape_from_trial")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _load_trial_config(trial_dir: Path):
    config_path = trial_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in trial directory: {trial_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _device(training_cfg):
    seed = int(training_cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_mape_maps(trial_dir: Path):
    logger = _configure_logger()
    logger.info("Loading trial from %s", trial_dir)

    cfg = _load_trial_config(trial_dir)
    data_cfg = cfg["data"]
    mask_cfg = cfg["mask"]
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    geometry_mask = DEFAULT_GEOMETRY_MASK.astype(bool).copy()
    _, missing_mask, observed_mask = build_observed_and_missing_masks(
        geometry_mask,
        mask_cfg["frac_missing"],
        mask_cfg["pattern"],
        mask_cfg["stride"],
        mask_cfg["axis"],
        seed=mask_cfg["seed"],
    )

    h, w = geometry_mask.shape
    levels_raw, _ = load_multiple_levels(data_cfg["multi_h5_paths"], h, w)
    train_idx, val_idx, test_idx = make_time_splits(
        levels_raw[0].shape[0],
        split_cfg["train_frac"],
        split_cfg["val_frac"],
    )
    z_levels, mu_l, sd_l = fit_transform_per_level_observed_data(levels_raw, train_idx, observed_mask)

    _, _, _, _, _, test_loader = build_datasets_and_loaders(
        "3d",
        z_levels,
        train_idx,
        val_idx,
        test_idx,
        observed_mask,
        missing_mask,
        geometry_mask,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    device = _device(training_cfg)
    logger.info("Using device: %s", device)

    model = Core3DInpaintNet(
        in_ch=model_cfg["in_ch"],
        base=model_cfg["base_channels"],
        dilations_xy=tuple(model_cfg["dilations_xy"]),
        kdepth=model_cfg["kdepth"],
        add_coords_3d=model_cfg["add_coords_3d"],
    ).to(device)

    state_dict_path = trial_dir / "model_state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Missing model_state_dict.pt in trial directory: {trial_dir}")
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    mu_t = torch.as_tensor(mu_l).view(1, 1, len(mu_l), 1, 1)
    sd_t = torch.as_tensor(sd_l).view(1, 1, len(sd_l), 1, 1)
    preds = []
    truths = []
    misses = []

    logger.info("Running inference on test set (%d timesteps)", len(test_idx))
    with torch.no_grad():
        for x, y, m_miss, _, _ in test_loader:
            x = x.to(device)
            y = y.to(device)
            m_miss = m_miss.to(device)
            y_hat = model(x).cpu()
            preds.append(sd_t * y_hat + mu_t)
            truths.append(sd_t * y.cpu() + mu_t)
            misses.append(m_miss.cpu())

    y_true = torch.cat(truths, dim=0)
    y_pred = torch.cat(preds, dim=0)
    miss_mask = torch.cat(misses, dim=0)

    per_level_pct = compute_per_level_percentage_maps(y_true, y_pred, miss_mask)
    level_titles = [Path(p).name for p in data_cfg["multi_h5_paths"]]

    fig, _ = plot_per_level_maps(
        per_level_pct["mape_maps"],
        titles=level_titles,
        colorbar_label="MAPE (%)",
        separate_colorbars=True,
    )

    out_png = trial_dir / "per_level_mape_maps.png"
    out_npz = trial_dir / "per_level_mape_maps.npz"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    np.savez(out_npz, mape_maps=per_level_pct["mape_maps"], mpse_maps=per_level_pct["mpse_maps"], count_maps=per_level_pct["count_maps"])

    logger.info("Saved MAPE subplot to %s", out_png)
    logger.info("Saved MAPE arrays to %s", out_npz)


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained 3D trial folder, run test predictions, and save per-level MAPE map subplot."
    )
    parser.add_argument(
        "--phase-trial-dir",
        required=True,
        help="Path to a single phase trial directory under experiments/field_reconstruction/results/.",
    )
    args = parser.parse_args()

    generate_mape_maps(Path(args.phase_trial_dir))


if __name__ == "__main__":
    main()
