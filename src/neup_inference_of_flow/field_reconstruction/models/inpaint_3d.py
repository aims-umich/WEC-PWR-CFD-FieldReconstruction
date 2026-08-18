import torch.nn as nn

from .common import add_3d_coordinate_channels


class ResidualDilatedBlock3D(nn.Module):
    """3D residual block with anisotropic dilation: depth uses 1; XY uses ``d_xy``."""

    def __init__(self, ch: int, d_xy: int = 1, kdepth: int = 3, groups: int = 8, dropout: float = 0.05):
        super().__init__()
        assert kdepth in (1, 3), "Use kdepth=1 or 3 to preserve depth alignment."
        pad_d = kdepth // 2
        pad_xy = d_xy
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=(kdepth, 3, 3), padding=(pad_d, pad_xy, pad_xy), dilation=(1, d_xy, d_xy))
        self.gn1 = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=(kdepth, 3, 3), padding=(pad_d, pad_xy, pad_xy), dilation=(1, d_xy, d_xy))
        self.gn2 = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
        self.drop = nn.Dropout3d(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = self.gn2(self.conv2(x))
        x = self.drop(x)
        return self.act(x + residual)


class Core3DInpaintNet(nn.Module):
    """3D inpainting network with observed-voxel copy-through on known cells."""

    def __init__(self, in_ch: int = 3, base: int = 64,
                 dilations_xy=(1, 2, 3, 3, 2, 1), kdepth: int = 3, add_coords_3d: bool = True):
        super().__init__()
        self.add_coords_3d = add_coords_3d
        stem_in = in_ch + (3 if add_coords_3d else 0)

        self.stem = nn.Sequential(
            nn.Conv3d(stem_in, base, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base, base, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualDilatedBlock3D(base, d_xy=d, kdepth=kdepth) for d in dilations_xy
        ])
        self.head = nn.Sequential(
            nn.Conv3d(base, base // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(base // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        v_obs = x[:, 0:1]
        m_obs = x[:, 1:2]
        m_geom = x[:, 2:3]
        m_miss = (m_geom - m_obs).clamp_(0, 1)

        xin = add_3d_coordinate_channels(x) if self.add_coords_3d else x
        h = self.stem(xin)
        h = self.blocks(h)
        pred = self.head(h)
        return v_obs + m_miss * pred
