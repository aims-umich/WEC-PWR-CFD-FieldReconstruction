import torch


def add_2d_coordinate_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized ``x``/``y`` coordinate channels to ``(B, C, H, W)`` inputs."""
    batch, _, height, width = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=x.device),
        torch.linspace(-1, 1, width, device=x.device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).expand(batch, -1, height, width)
    return torch.cat([x, coords], dim=1)



def add_3d_coordinate_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized ``z``/``y``/``x`` coordinate channels to ``(B, C, D, H, W)`` inputs."""
    batch, _, depth, height, width = x.shape
    zz = torch.linspace(-1, 1, depth, device=x.device).view(1, 1, depth, 1, 1).expand(batch, 1, depth, height, width)
    yy = torch.linspace(-1, 1, height, device=x.device).view(1, 1, 1, height, 1).expand(batch, 1, depth, height, width)
    xx = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, 1, width).expand(batch, 1, depth, height, width)
    return torch.cat([x, zz, yy, xx], dim=1)
