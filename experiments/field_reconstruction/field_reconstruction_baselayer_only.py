from neup_inference_of_flow.field_reconstruction import (
    DataPathsConfig,
    MaskConfig,
    Model2DConfig,
    SingleLevelExperimentConfig,
    SplitConfig,
    TrainingConfig,
    run_single_level_experiment,
)

CONFIG = SingleLevelExperimentConfig(
    data=DataPathsConfig(
        single_h5_path="../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_01_Base.h5",
    ),
    mask=MaskConfig(frac_missing=0.50, pattern="checkerboard", stride=None, axis="row", seed=42),
    split=SplitConfig(seed=123),
    model=Model2DConfig(base_channels=96, dilations=(1, 2, 3, 4, 3, 2), add_coords=True),
    training=TrainingConfig(
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=None,
        use_amp=False,
        patience=4,
        lr_factor=0.5,
        seed=1337,
        batch_size=64,
        num_workers=0,
    ),
    plot=True,
)


if __name__ == "__main__":
    run_single_level_experiment(CONFIG)
