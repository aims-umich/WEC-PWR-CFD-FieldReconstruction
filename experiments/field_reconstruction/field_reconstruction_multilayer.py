from neup_inference_of_flow.field_reconstruction import (
    DataPathsConfig,
    MaskConfig,
    Model2DConfig,
    MultiLevel2DExperimentConfig,
    SplitConfig,
    TrainingConfig,
    run_multilevel_2d_experiment,
)

CONFIG = MultiLevel2DExperimentConfig(
    data=DataPathsConfig(
        multi_h5_paths=(
            "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_01_Base.h5",
            "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_02_Layer1.h5",
            "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_03_Layer2.h5",
            "../postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5/plane_04_Layer3.h5",
        ),
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
    run_multilevel_2d_experiment(CONFIG)
