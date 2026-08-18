import numpy as np
import torch



def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def compute_missing_region_metrics(y_true, y_pred, miss_mask, eps=1e-6):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    miss_mask = _to_numpy(miss_mask).astype(bool)

    diff = (y_pred - y_true)[miss_mask]
    true_vals = y_true[miss_mask]
    n = diff.size
    if n == 0:
        return {"mae": np.nan, "mape": np.nan, "r2": np.nan, "count": 0, "errors": diff}

    abs_err = np.abs(diff)
    mae = float(abs_err.mean())
    mape = float(np.mean(abs_err / np.clip(np.abs(true_vals), eps, None)) * 100.0)
    mean_y = float(true_vals.mean())
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true_vals - mean_y) ** 2))
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-12)))
    return {"mae": mae, "mape": mape, "r2": r2, "count": int(n), "errors": diff}



def compute_per_cell_error_maps(y_true, y_pred, miss_mask):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    miss_mask = _to_numpy(miss_mask).astype(bool)

    if y_true.ndim == 4:
        y_true = np.squeeze(y_true, axis=1)
        y_pred = np.squeeze(y_pred, axis=1)
        miss_mask = np.squeeze(miss_mask, axis=1)

    diff = (y_pred - y_true) * miss_mask
    abs_err = np.abs(diff)
    count_grid = miss_mask.sum(axis=0).astype(np.float64)
    sum_abs_grid = abs_err.sum(axis=0)
    signed_sum_grid = diff.sum(axis=0)

    mae_map = np.full(count_grid.shape, np.nan, dtype=np.float64)
    mean_err_map = np.full(count_grid.shape, np.nan, dtype=np.float64)
    has_data = count_grid > 0
    mae_map[has_data] = sum_abs_grid[has_data] / count_grid[has_data]
    mean_err_map[has_data] = signed_sum_grid[has_data] / count_grid[has_data]
    return {
        "mae_map": mae_map,
        "mean_err_map": mean_err_map,
        "count_map": count_grid,
        "sum_abs_map": sum_abs_grid,
        "signed_sum_map": signed_sum_grid,
    }



def compute_per_level_metrics(y_true, y_pred, miss_mask, eps=1e-6):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    miss_mask = _to_numpy(miss_mask).astype(bool)

    if y_true.ndim != 5:
        raise ValueError("Expected y_true/y_pred/miss_mask with shape (B,L,1,H,W) or (B,1,L,H,W).")

    if y_true.shape[1] == 1:
        y_true = np.transpose(y_true, (0, 2, 1, 3, 4))
        y_pred = np.transpose(y_pred, (0, 2, 1, 3, 4))
        miss_mask = np.transpose(miss_mask, (0, 2, 1, 3, 4))

    y_true = np.squeeze(y_true, axis=2)
    y_pred = np.squeeze(y_pred, axis=2)
    miss_mask = np.squeeze(miss_mask, axis=2)

    num_levels = y_true.shape[1]
    metrics = []
    mae_maps = []
    mean_err_maps = []
    err_samples = []
    for level in range(num_levels):
        level_metrics = compute_missing_region_metrics(y_true[:, level], y_pred[:, level], miss_mask[:, level], eps=eps)
        level_maps = compute_per_cell_error_maps(y_true[:, level], y_pred[:, level], miss_mask[:, level])
        metrics.append(level_metrics)
        mae_maps.append(level_maps["mae_map"])
        mean_err_maps.append(level_maps["mean_err_map"])
        err_samples.append(level_metrics["errors"])

    return {
        "metrics": metrics,
        "mae": np.array([m["mae"] for m in metrics], dtype=np.float64),
        "mape": np.array([m["mape"] for m in metrics], dtype=np.float64),
        "r2": np.array([m["r2"] for m in metrics], dtype=np.float64),
        "counts": np.array([m["count"] for m in metrics], dtype=np.float64),
        "mae_maps": np.stack(mae_maps, axis=0),
        "mean_err_maps": np.stack(mean_err_maps, axis=0),
        "error_samples": err_samples,
    }



def compute_per_level_percentage_maps(y_true, y_pred, miss_mask, eps=1e-6):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    miss_mask = _to_numpy(miss_mask).astype(bool)

    if y_true.shape[1] == 1:
        y_true = np.transpose(y_true, (0, 2, 1, 3, 4))
        y_pred = np.transpose(y_pred, (0, 2, 1, 3, 4))
        miss_mask = np.transpose(miss_mask, (0, 2, 1, 3, 4))

    y_true = np.squeeze(y_true, axis=2)
    y_pred = np.squeeze(y_pred, axis=2)
    miss_mask = np.squeeze(miss_mask, axis=2)

    diff = (y_pred - y_true) * miss_mask
    denom = np.clip(np.abs(y_true), eps, None)
    abs_pct = np.abs(diff) / denom * miss_mask
    signed_pct = diff / denom * miss_mask
    count_grid = miss_mask.sum(axis=0).astype(np.float64)

    mape_maps = np.full(count_grid.shape, np.nan, dtype=np.float64)
    mpse_maps = np.full(count_grid.shape, np.nan, dtype=np.float64)
    has_data = count_grid > 0
    for level in range(count_grid.shape[0]):
        mask = has_data[level]
        mape_maps[level, mask] = 100.0 * abs_pct[:, level].sum(axis=0)[mask] / count_grid[level, mask]
        mpse_maps[level, mask] = 100.0 * signed_pct[:, level].sum(axis=0)[mask] / count_grid[level, mask]

    return {"mape_maps": mape_maps, "mpse_maps": mpse_maps, "count_maps": count_grid}
