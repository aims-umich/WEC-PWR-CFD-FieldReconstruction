import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import MultiLevel2DExperimentConfig, MultiLevel3DExperimentConfig, SingleLevelExperimentConfig
from .datasets import build_datasets_and_loaders
from .io import load_first_matching_txhxw, load_multiple_levels
from .masks import DEFAULT_GEOMETRY_MASK, build_observed_and_missing_masks
from .metrics import compute_missing_region_metrics, compute_per_cell_error_maps, compute_per_level_metrics
from .models import Core3DInpaintNet, CoreInpaintNet
from .plots import plot_heatmap, plot_mask_composite, plot_per_level_error_histograms, plot_per_level_maps, plot_training_history
from .preprocessing import fit_transform_observed_data, fit_transform_per_level_observed_data, make_time_splits
from .training import build_optimizer_scheduler_scaler, masked_mse_2d, masked_mse_3d, masked_mse_multilevel, run_epoch


class MultiLevelWrapper(nn.Module):
    def __init__(self, single_level_model: nn.Module):
        super().__init__()
        self.m = single_level_model

    def forward(self, x):
        b, l, c, h, w = x.shape
        y = self.m(x.view(b * l, c, h, w))
        return y.view(b, l, 1, h, w)


def _level_titles_from_paths(paths):
    return [os.path.basename(p) for p in list(paths)]



def _device_and_seed(training_config):
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _mask_triplet(mask_config):
    geometry_mask = DEFAULT_GEOMETRY_MASK.astype(bool).copy()
    _, missing_mask, observed_mask = build_observed_and_missing_masks(
        geometry_mask,
        mask_config.frac_missing,
        mask_config.pattern,
        mask_config.stride,
        mask_config.axis,
        seed=mask_config.seed,
    )
    return geometry_mask, missing_mask, observed_mask



def _split_indices(num_timesteps, split_config):
    return make_time_splits(num_timesteps, split_config.train_frac, split_config.val_frac)



def run_single_level_experiment(config: SingleLevelExperimentConfig):
    geometry_mask, missing_mask, observed_mask = _mask_triplet(config.mask)
    h, w = geometry_mask.shape
    data, ds_path = load_first_matching_txhxw(config.data.single_h5_path, h, w)
    train_idx, val_idx, test_idx = _split_indices(data.shape[0], config.split)
    z, mu, sd = fit_transform_observed_data(data, train_idx, observed_mask)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_datasets_and_loaders(
        "2d", z, train_idx, val_idx, test_idx, observed_mask, missing_mask, geometry_mask,
        batch_size=config.training.batch_size, num_workers=config.training.num_workers, pin_memory=True, drop_last=False,
    )
    device = _device_and_seed(config.training)
    model = CoreInpaintNet(
        in_ch=config.model.in_ch,
        base=config.model.base_channels,
        dilations=config.model.dilations,
        add_coords=config.model.add_coords,
    ).to(device)
    optimizer, scheduler, scaler = build_optimizer_scheduler_scaler(model, config.training.to_hp(), device)
    history = {"train": [], "val": []}
    best_state, best_val = None, float("inf")
    for _epoch in range(1, config.training.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, device, True, masked_mse_2d, grad_clip=config.training.grad_clip, amp_enabled=config.training.use_amp)
        va = run_epoch(model, val_loader, optimizer, scaler, device, False, masked_mse_2d, grad_clip=config.training.grad_clip, amp_enabled=config.training.use_amp)
        history["train"].append(tr)
        history["val"].append(va)
        scheduler.step(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    preds = []
    truths = []
    misses = []
    model.eval()
    with torch.no_grad():
        for x, y, m_miss, _, _ in test_loader:
            x = x.to(device)
            y = y.to(device)
            m_miss = m_miss.to(device)
            y_hat = model(x)
            preds.append((sd * y_hat + mu).cpu())
            truths.append((sd * y + mu).cpu())
            misses.append(m_miss.cpu())
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(truths, dim=0)
    miss_mask_t = torch.cat(misses, dim=0)
    scalar_metrics = compute_missing_region_metrics(y_true, y_pred, miss_mask_t)
    cell_maps = compute_per_cell_error_maps(y_true, y_pred, miss_mask_t)
    if config.plot:
        plot_mask_composite(geometry_mask, observed_mask, missing_mask)
        plot_training_history(history, title="Training curves")
        plot_heatmap(cell_maps["mae_map"], "Per-cell MAE on TEST", "MAE (kg/s)")
    return {
        "config": asdict(config),
        "dataset_path": ds_path,
        "history": history,
        "metrics": scalar_metrics,
        "cell_maps": cell_maps,
        "splits": (train_idx, val_idx, test_idx),
        "model": model,
        "datasets": (train_ds, val_ds, test_ds),
    }



def run_multilevel_2d_experiment(config: MultiLevel2DExperimentConfig):
    geometry_mask, missing_mask, observed_mask = _mask_triplet(config.mask)
    h, w = geometry_mask.shape
    levels_raw, ds_paths = load_multiple_levels(config.data.multi_h5_paths, h, w)
    train_idx, val_idx, test_idx = _split_indices(levels_raw[0].shape[0], config.split)
    z_levels, mu_l, sd_l = fit_transform_per_level_observed_data(levels_raw, train_idx, observed_mask)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_datasets_and_loaders(
        "multilevel_2d", z_levels, train_idx, val_idx, test_idx, observed_mask, missing_mask, geometry_mask,
        batch_size=config.training.batch_size, num_workers=config.training.num_workers, pin_memory=True, drop_last=False,
    )
    device = _device_and_seed(config.training)
    core = CoreInpaintNet(in_ch=config.model.in_ch, base=config.model.base_channels, dilations=config.model.dilations, add_coords=config.model.add_coords).to(device)
    model = MultiLevelWrapper(core).to(device)
    optimizer, scheduler, scaler = build_optimizer_scheduler_scaler(model, config.training.to_hp(), device)
    history = {"train": [], "val": []}
    best_state, best_val = None, float("inf")
    for _epoch in range(1, config.training.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, device, True, masked_mse_multilevel, grad_clip=config.training.grad_clip, amp_enabled=config.training.use_amp)
        va = run_epoch(model, val_loader, optimizer, scaler, device, False, masked_mse_multilevel, grad_clip=config.training.grad_clip, amp_enabled=config.training.use_amp)
        history["train"].append(tr); history["val"].append(va)
        scheduler.step(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    mu_t = torch.as_tensor(mu_l).view(1, len(mu_l), 1, 1, 1)
    sd_t = torch.as_tensor(sd_l).view(1, len(sd_l), 1, 1, 1)
    preds = []; truths = []; misses = []
    model.eval()
    with torch.no_grad():
        for x, y, m_miss, _, _ in test_loader:
            x = x.to(device); y = y.to(device); m_miss = m_miss.to(device)
            y_hat = model(x).cpu()
            preds.append(sd_t * y_hat + mu_t)
            truths.append(sd_t * y.cpu() + mu_t)
            misses.append(m_miss.cpu())
    per_level = compute_per_level_metrics(torch.cat(truths, dim=0), torch.cat(preds, dim=0), torch.cat(misses, dim=0))
    if config.plot:
        plot_training_history(history, title="Training curves (multi-level)")
        level_titles = _level_titles_from_paths(config.data.multi_h5_paths)
        plot_per_level_maps(per_level["mae_maps"], titles=level_titles, colorbar_label="MAE (kg/s)")
    return {"config": asdict(config), "dataset_paths": ds_paths, "history": history, "per_level": per_level, "model": model, "datasets": (train_ds, val_ds, test_ds)}



def run_multilevel_3d_experiment(config: MultiLevel3DExperimentConfig, results_dir_name=None, output_dir=None):
    geometry_mask, missing_mask, observed_mask = _mask_triplet(config.mask)
    h, w = geometry_mask.shape
    levels_raw, ds_paths = load_multiple_levels(config.data.multi_h5_paths, h, w)
    train_idx, val_idx, test_idx = _split_indices(levels_raw[0].shape[0], config.split)
    z_levels, mu_l, sd_l = fit_transform_per_level_observed_data(levels_raw, train_idx, observed_mask)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_datasets_and_loaders(
        "3d", z_levels, train_idx, val_idx, test_idx, observed_mask, missing_mask, geometry_mask,
        batch_size=config.training.batch_size, num_workers=config.training.num_workers, pin_memory=True, drop_last=False,
    )
    device = _device_and_seed(config.training)
    print(f"[run_multilevel_3d_experiment] Selected device: {device}")
    if device.type == "cuda":
        print(f"[run_multilevel_3d_experiment] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[run_multilevel_3d_experiment] CUDA unavailable; training on CPU.")
    test_frac = 1.0 - config.split.train_frac - config.split.val_frac
    print(
        f"[run_multilevel_3d_experiment] Split fractions: "
        f"train={config.split.train_frac:.2f}, val={config.split.val_frac:.2f}, test={test_frac:.2f}"
    )
    print(
        f"[run_multilevel_3d_experiment] Split counts (timesteps): "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, total={levels_raw[0].shape[0]}"
    )
    model = Core3DInpaintNet(
        in_ch=config.model.in_ch,
        base=config.model.base_channels,
        dilations_xy=config.model.dilations_xy,
        kdepth=config.model.kdepth,
        add_coords_3d=config.model.add_coords_3d,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.lr_factor,
        patience=config.training.patience,
        verbose=True,
    )
    scaler = None
    history = {"train": [], "val": []}
    best_state, best_val = None, float("inf")
    for _epoch in range(1, config.training.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, device, True, masked_mse_3d, grad_clip=None, amp_enabled=False)
        va = run_epoch(model, val_loader, optimizer, scaler, device, False, masked_mse_3d, grad_clip=None, amp_enabled=False)
        history["train"].append(tr); history["val"].append(va)
        scheduler.step(va)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[run_multilevel_3d_experiment] Epoch {_epoch:03d}/{config.training.epochs} "
            f"| train_loss={tr:.6e} | val_loss={va:.6e} | lr={current_lr:.6e}"
        )
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[run_multilevel_3d_experiment] New best val_loss={best_val:.6e} at epoch {_epoch:03d}")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        print(f"[run_multilevel_3d_experiment] Loaded best checkpoint with val_loss={best_val:.6e}")

    mu_t = torch.as_tensor(mu_l).view(1, 1, len(mu_l), 1, 1)
    sd_t = torch.as_tensor(sd_l).view(1, 1, len(sd_l), 1, 1)
    preds = []; truths = []; misses = []
    model.eval()
    with torch.no_grad():
        for x, y, m_miss, _, _ in test_loader:
            x = x.to(device); y = y.to(device); m_miss = m_miss.to(device)
            y_hat = model(x).cpu()
            preds.append(sd_t * y_hat + mu_t)
            truths.append(sd_t * y.cpu() + mu_t)
            misses.append(m_miss.cpu())
    per_level = compute_per_level_metrics(torch.cat(truths, dim=0), torch.cat(preds, dim=0), torch.cat(misses, dim=0))
    repo_root = Path(__file__).resolve().parents[3]
    if output_dir is not None and results_dir_name is not None:
        raise ValueError("Provide only one of output_dir and results_dir_name.")
    if output_dir is not None:
        results_dir = Path(output_dir).expanduser().resolve()
    elif results_dir_name is not None:
        results_dir = repo_root / "experiments" / "field_reconstruction" / "results" / str(results_dir_name)
    else:
        raise ValueError("output_dir or results_dir_name must be provided.")
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with (results_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (results_dir / "dataset_paths.json").open("w", encoding="utf-8") as f:
        json.dump(list(ds_paths), f, indent=2)

    np.savez(
        results_dir / "per_level_arrays.npz",
        mae=per_level["mae"],
        mape=per_level["mape"],
        r2=per_level["r2"],
        counts=per_level["counts"],
        mae_maps=per_level["mae_maps"],
        mean_err_maps=per_level["mean_err_maps"],
    )
    np.save(results_dir / "error_samples.npy", np.array(per_level["error_samples"], dtype=object), allow_pickle=True)

    per_level_summary = {
        "mae": np.asarray(per_level["mae"]).tolist(),
        "mape": np.asarray(per_level["mape"]).tolist(),
        "r2": np.asarray(per_level["r2"]).tolist(),
        "counts": np.asarray(per_level["counts"]).tolist(),
    }
    with (results_dir / "per_level_summary.json").open("w", encoding="utf-8") as f:
        json.dump(per_level_summary, f, indent=2)

    if config.plot:
        fig_hist, _ = plot_training_history(history, title="Training curves (3D CNN)")
        level_titles = _level_titles_from_paths(config.data.multi_h5_paths)
        fig_maps, _ = plot_per_level_maps(
            per_level["mae_maps"],
            titles=level_titles,
            colorbar_label="MAE (kg/s)",
            separate_colorbars=True,
        )
        fig_error_hists, _ = plot_per_level_error_histograms(per_level["error_samples"], titles=level_titles)
        fig_hist.savefig(results_dir / "training_curves_3d.png", dpi=200, bbox_inches="tight")
        fig_maps.savefig(results_dir / "per_level_mae_maps.png", dpi=200, bbox_inches="tight")
        fig_error_hists.savefig(results_dir / "per_level_error_histograms.png", dpi=200, bbox_inches="tight")
    torch.save(model.state_dict(), results_dir / "model_state_dict.pt")
    print(f"[run_multilevel_3d_experiment] Saved results to: {results_dir}")
    return {
        "config": asdict(config),
        "dataset_paths": ds_paths,
        "history": history,
        "per_level": per_level,
        "model": model,
        "datasets": (train_ds, val_ds, test_ds),
        "results_dir": str(results_dir),
    }
