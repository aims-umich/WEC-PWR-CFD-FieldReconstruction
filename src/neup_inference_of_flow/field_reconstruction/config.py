from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DataPathsConfig:
    single_h5_path: str | None = None
    multi_h5_paths: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class MaskConfig:
    frac_missing: float = 0.50
    pattern: str = "checkerboard"
    stride: int | tuple[int, int] | None = None
    axis: str = "row"
    seed: int = 42


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.45
    val_frac: float = 0.1
    seed: int = 123


@dataclass(frozen=True)
class Model2DConfig:
    in_ch: int = 3
    base_channels: int = 96
    dilations: tuple[int, ...] = (1, 2, 3, 4, 3, 2)
    add_coords: bool = True


@dataclass(frozen=True)
class Model3DConfig:
    in_ch: int = 3
    base_channels: int = 96
    dilations_xy: tuple[int, ...] = (1, 2, 3, 3, 2, 1)
    kdepth: int = 3
    add_coords_3d: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = None
    use_amp: bool = False
    patience: int = 4
    lr_factor: float = 0.5
    seed: int = 1337
    batch_size: int = 64
    num_workers: int = 0

    def to_hp(self) -> dict:
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "use_amp": self.use_amp,
            "patience": self.patience,
            "lr_factor": self.lr_factor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class SingleLevelExperimentConfig:
    data: DataPathsConfig
    mask: MaskConfig = field(default_factory=MaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: Model2DConfig = field(default_factory=Model2DConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plot: bool = True


@dataclass(frozen=True)
class MultiLevel2DExperimentConfig:
    data: DataPathsConfig
    mask: MaskConfig = field(default_factory=MaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: Model2DConfig = field(default_factory=Model2DConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plot: bool = True


@dataclass(frozen=True)
class MultiLevel3DExperimentConfig:
    data: DataPathsConfig
    mask: MaskConfig = field(default_factory=MaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: Model3DConfig = field(default_factory=Model3DConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plot: bool = True
