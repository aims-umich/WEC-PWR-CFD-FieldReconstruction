import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



def broadcast_masks_to_levels(observed_mask_2d, missing_mask_2d, geometry_mask_2d, num_levels, layout="multilevel_2d"):
    """Broadcast 2D masks to the requested multi-level tensor layout.

    Parameters
    ----------
    observed_mask_2d, missing_mask_2d, geometry_mask_2d:
        2D mask arrays or tensors with shape ``(H, W)`` or ``(1, H, W)``.
    num_levels:
        Number of axial levels ``L``.
    layout:
        ``"multilevel_2d"`` returns masks shaped ``(L, 1, H, W)``.
        ``"3d"`` returns masks shaped ``(1, L, H, W)``.
    """
    obs = torch.as_tensor(observed_mask_2d, dtype=torch.float32)
    miss = torch.as_tensor(missing_mask_2d, dtype=torch.float32)
    geom = torch.as_tensor(geometry_mask_2d, dtype=torch.float32)

    if obs.ndim == 2:
        obs = obs.unsqueeze(0)
    if miss.ndim == 2:
        miss = miss.unsqueeze(0)
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)

    if layout == "multilevel_2d":
        return (
            obs.unsqueeze(0).repeat(num_levels, 1, 1, 1),
            miss.unsqueeze(0).repeat(num_levels, 1, 1, 1),
            geom.unsqueeze(0).repeat(num_levels, 1, 1, 1),
        )
    if layout == "3d":
        return (
            obs.unsqueeze(0).repeat(1, num_levels, 1, 1),
            miss.unsqueeze(0).repeat(1, num_levels, 1, 1),
            geom.unsqueeze(0).repeat(1, num_levels, 1, 1),
        )
    raise ValueError('layout must be "multilevel_2d" or "3d"')



def make_model_inputs_from_values(values, observed_mask, geometry_mask, channel_dim=0):
    """Build model inputs by concatenating observed values, observed mask, and geometry mask."""
    x_vals = values * observed_mask
    return torch.cat([x_vals, observed_mask, geometry_mask], dim=channel_dim)



class Inpaint2DDataset(Dataset):
    """Single-level inpainting dataset.

    Per sample returns:
    - ``x`` with shape ``(3, H, W)`` and channel order
      ``[values*observed_mask, observed_mask, geometry_mask]``
    - ``y`` with shape ``(1, H, W)``
    - ``m_missing``, ``m_observed``, ``m_geom`` each with shape ``(1, H, W)``
    """

    def __init__(self, z_scaled, idxs, observed_mask, missing_mask, geometry_mask):
        self.z = np.asarray(z_scaled)
        self.idxs = np.asarray(idxs)
        self.m_obs = torch.as_tensor(observed_mask, dtype=torch.float32)
        self.m_miss = torch.as_tensor(missing_mask, dtype=torch.float32)
        self.m_geom = torch.as_tensor(geometry_mask, dtype=torch.float32)
        if self.m_obs.ndim == 2:
            self.m_obs = self.m_obs.unsqueeze(0)
        if self.m_miss.ndim == 2:
            self.m_miss = self.m_miss.unsqueeze(0)
        if self.m_geom.ndim == 2:
            self.m_geom = self.m_geom.unsqueeze(0)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = int(self.idxs[i])
        frame = torch.from_numpy(self.z[t]).float().unsqueeze(0)
        x = make_model_inputs_from_values(frame, self.m_obs, self.m_geom, channel_dim=0)
        y = frame * self.m_geom
        return x, y, self.m_miss, self.m_obs, self.m_geom



class MultiLevelInpaint2DDataset(Dataset):
    """Multi-level 2D inpainting dataset.

    Per sample returns:
    - ``x`` with shape ``(L, 3, H, W)`` and per-level channel order
      ``[values*observed_mask, observed_mask, geometry_mask]``
    - ``y`` with shape ``(L, 1, H, W)``
    - ``m_missing``, ``m_observed``, ``m_geom`` each with shape ``(L, 1, H, W)``
    """

    def __init__(self, z_levels, idxs, observed_mask_2d, missing_mask_2d, geometry_mask_2d):
        self.z = torch.stack([torch.from_numpy(np.asarray(z)).float() for z in z_levels], dim=1)
        self.idxs = np.asarray(idxs)
        num_levels = len(z_levels)
        self.m_obs, self.m_miss, self.m_geom = broadcast_masks_to_levels(
            observed_mask_2d,
            missing_mask_2d,
            geometry_mask_2d,
            num_levels,
            layout="multilevel_2d",
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = int(self.idxs[i])
        values = self.z[t].unsqueeze(1)
        x = make_model_inputs_from_values(values, self.m_obs, self.m_geom, channel_dim=1)
        y = values * self.m_geom
        return x, y, self.m_miss, self.m_obs, self.m_geom



class Inpaint3DDataset(Dataset):
    """3D inpainting dataset.

    Per sample returns:
    - ``x`` with shape ``(3, L, H, W)`` and channel order
      ``[values*observed_mask, observed_mask, geometry_mask]``
    - ``y`` with shape ``(1, L, H, W)``
    - ``m_missing``, ``m_observed``, ``m_geom`` each with shape ``(1, L, H, W)``
    """

    def __init__(self, z_levels, idxs, observed_mask_2d, missing_mask_2d, geometry_mask_2d):
        self.z = torch.from_numpy(np.stack(z_levels, axis=1)).float()
        self.idxs = np.asarray(idxs)
        num_levels = len(z_levels)
        self.m_obs, self.m_miss, self.m_geom = broadcast_masks_to_levels(
            observed_mask_2d,
            missing_mask_2d,
            geometry_mask_2d,
            num_levels,
            layout="3d",
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = int(self.idxs[i])
        values = self.z[t].unsqueeze(0)
        x = make_model_inputs_from_values(values, self.m_obs, self.m_geom, channel_dim=0)
        y = values * self.m_geom
        return x, y, self.m_miss, self.m_obs, self.m_geom



def build_datasets_and_loaders(
    kind,
    data,
    train_idx,
    val_idx,
    test_idx,
    observed_mask,
    missing_mask,
    geometry_mask,
    batch_size,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
):
    dataset_map = {
        "2d": Inpaint2DDataset,
        "multilevel_2d": MultiLevelInpaint2DDataset,
        "3d": Inpaint3DDataset,
    }
    if kind not in dataset_map:
        raise ValueError('kind must be "2d", "multilevel_2d", or "3d"')

    dataset_cls = dataset_map[kind]
    train_ds = dataset_cls(data, train_idx, observed_mask, missing_mask, geometry_mask)
    val_ds = dataset_cls(data, val_idx, observed_mask, missing_mask, geometry_mask)
    test_ds = dataset_cls(data, test_idx, observed_mask, missing_mask, geometry_mask)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=drop_last)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
