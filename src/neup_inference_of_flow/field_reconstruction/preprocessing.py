import numpy as np


def make_time_splits(num_timesteps, train_frac=0.8, val_frac=0.1):
    ordered_idx = np.arange(num_timesteps)
    n_train = int(train_frac * num_timesteps)
    n_val = int(val_frac * num_timesteps)
    train_idx = ordered_idx[:n_train]
    val_idx = ordered_idx[n_train:n_train + n_val]
    test_idx = ordered_idx[n_train + n_val:]
    return train_idx, val_idx, test_idx



def fit_observed_scaler(data, train_idx, observed_mask, eps=1e-8):
    train_vals = data[train_idx][:, observed_mask].reshape(-1)
    mu = float(train_vals.mean())
    sd_raw = float(train_vals.std())
    sd = sd_raw if sd_raw > eps else 1.0
    return mu, sd



def transform_observed_data(data, mu, sd):
    z = (data - mu) / sd
    return z, mu, sd



def fit_transform_observed_data(data, train_idx, observed_mask, eps=1e-8):
    mu, sd = fit_observed_scaler(data, train_idx, observed_mask, eps=eps)
    return transform_observed_data(data, mu, sd)



def fit_per_level_observed_scalers(levels_raw, train_idx, observed_mask, eps=1e-8):
    mu_levels = []
    sd_levels = []
    for level_array in levels_raw:
        mu, sd = fit_observed_scaler(level_array, train_idx, observed_mask, eps=eps)
        mu_levels.append(mu)
        sd_levels.append(sd)
    return np.array(mu_levels, dtype=np.float32), np.array(sd_levels, dtype=np.float32)



def transform_per_level_observed_data(levels_raw, mu_levels, sd_levels):
    z_levels = []
    for level_array, mu, sd in zip(levels_raw, mu_levels, sd_levels):
        z_levels.append((level_array - mu) / sd)
    return z_levels, mu_levels, sd_levels



def fit_transform_per_level_observed_data(levels_raw, train_idx, observed_mask, eps=1e-8):
    mu_levels, sd_levels = fit_per_level_observed_scalers(levels_raw, train_idx, observed_mask, eps=eps)
    return transform_per_level_observed_data(levels_raw, mu_levels, sd_levels)



def observed_mean_std(data, idx, observed_mask):
    vals = data[idx][:, observed_mask].reshape(-1)
    return float(vals.mean()), float(vals.std())
