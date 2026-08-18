import torch
import torch.nn as nn

from .common import add_2d_coordinate_channels


class ResidualDilatedBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, groups: int = 8, dropout: float = 0.05):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = self.gn2(self.conv2(x))
        x = self.drop(x)
        return self.act(x + residual)


class CoreInpaintNet(nn.Module):
    """2D inpainting network with observed-value copy-through on known cells."""

    def __init__(self, in_ch: int = 3, base: int = 64,
                 dilations=(1, 2, 3, 4, 3, 2), add_coords: bool = True):
        super().__init__()
        self.add_coords = add_coords
        stem_in = in_ch + (2 if add_coords else 0)

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualDilatedBlock(base, d) for d in dilations])
        self.head = nn.Sequential(
            nn.Conv2d(base, base // 2, 1),
            nn.SiLU(),
            nn.Conv2d(base // 2, 1, 1),
        )

    def forward(self, x):
        v_obs = x[:, 0:1]
        m_obs = x[:, 1:2]
        m_geom = x[:, 2:3]
        m_miss = (m_geom - m_obs).clamp_(0, 1)

        xin = add_2d_coordinate_channels(x) if self.add_coords else x
        h = self.stem(xin)
        h = self.blocks(h)
        pred_missing = self.head(h)
        return v_obs + m_miss * pred_missing
