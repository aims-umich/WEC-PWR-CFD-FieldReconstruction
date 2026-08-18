# Publication field-reconstruction workflow

This directory contains the field-reconstruction experiments associated with
the paper. The **publication configuration** is the joint, four-level 3D CNN in
[`publication_configuration.json`](publication_configuration.json); it is the
only configuration designated here for reproducing the reported result. Run
commands from the repository root unless an absolute script path is used. The
entry point resolves configured inputs and outputs relative to the configuration
file and resolves CLI path overrides relative to the repository, not the
caller's current working directory.

## 1. Generate the four inputs

Begin with the batch CSV exports in the publication case's `raw_csv` directory:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --batches-dir postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/raw_csv \
  --t-min 10 --t-max 15
```

The pipeline verifies identical batch time vectors, combines the batches,
orders the 193 assembly monitors by axial layer, and writes one HDF5 file per
plane below the case's `hdf5/` directory. The publication experiment consumes:

1. `plane_01_Base.h5`
2. `plane_02_Layer1.h5`
3. `plane_03_Layer2.h5`
4. `plane_04_Layer3.h5`

If that case directory already contains generated outputs, add `--overwrite`.
The `[10, 15]` s selection supplies the paper's transient interval; the
chronological split assigns 45%/10%/45% to train/validation/test (4,500/1,000/
4,500 snapshots for the 10,000-snapshot publication data).

## 2. Verify the HDF5 contract

Every file has exactly the datasets created by the processing pipeline:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `time` | `(T,)` | Physical time for each aligned snapshot |
| `data` | `(T, 15, 15)` | Assembly mass flow, with the 193 valid core cells filled and the 32 cells outside the core set to zero |

For the publication inputs, `T = 10,000`, so the exact shapes are `(10000,)`
and `(10000, 15, 15)`. All four `time` arrays must be aligned and all four
`data` arrays must have the same `T`. The model loader selects the three-
dimensional `data` dataset; it then stacks the levels into model tensors with
input shape `(batch, 3, 4, 15, 15)` (observed value, observed mask, geometry
mask) and target/mask shape `(batch, 1, 4, 15, 15)`.

An optional contract check is:

```bash
python - <<'PY'
import h5py
from pathlib import Path

root = Path("postprocessing/porous_full_core/cubic_k-eps/FApitch_over14/hdf5")
for path in [root / "plane_01_Base.h5", root / "plane_02_Layer1.h5",
             root / "plane_03_Layer2.h5", root / "plane_04_Layer3.h5"]:
    with h5py.File(path) as h5:
        print(path.name, {name: value.shape for name, value in h5.items()})
PY
```

## 3. Run the publication configuration

The canonical command, explicitly named the **publication configuration**, is:

```bash
python experiments/field_reconstruction/field_reconstruction_multilayer_3DCNN.py
```

It loads `publication_configuration.json`, including the checkerboard 50%
missing mask, chronological 0.45/0.10/0.45 split, final 3D-CNN architecture,
and tuned training hyperparameters. To use equivalent data elsewhere, repeat
`--input-path PATH` exactly four times. `--output-dir PATH` changes the output
location. Repeat `--set SECTION.KEY=JSON_VALUE` for controlled overrides, for
example `--set training.epochs=10 --set training.batch_size=32`. CLI input and
output paths are repository-relative; paths inside another configuration are
relative to that configuration file.

## 4. Locate the saved artifacts

The default run is written to
`experiments/field_reconstruction/results/publication_configuration/`:

| Artifact | Purpose |
| --- | --- |
| `config.json` | Fully resolved effective experiment configuration |
| `dataset_paths.json` | HDF5 dataset selected in each input (`/data`) |
| `model_state_dict.pt` | Best-validation model checkpoint used for testing |
| `per_level_summary.json` | Plane-wise MAE, MAPE, R², and hidden-cell counts |
| `per_level_arrays.npz` | Metrics plus plane-wise MAE and signed-error maps |
| `error_samples.npy` | Per-plane hidden-cell errors used by histograms |
| `history.json` | Per-epoch training and validation loss history |
| `training_curves_3d.png` | Loss-history plot |
| `per_level_mae_maps.png` | Four plane-wise spatial MAE plots |
| `per_level_error_histograms.png` | Four plane-wise error distributions |

## 5. Connect artifacts to the paper

For `tab:field-recon-results`, read the four entries in each of `mae`, `mape`,
and `r2` from `per_level_summary.json`, in Base/Layer1/Layer2/Layer3 order. The
publication table reports respectively MAE `2.6062, 0.0641, 0.0120, 0.0043`
kg/s, MAPE `3.3473, 0.0825, 0.0154, 0.0055`%, and R² `0.7593, 0.9966,
0.9986, 0.9982`. A newly reproduced run should be compared with these values;
do not silently treat a run with overrides as the publication run.

For `fig:field-recon-flow`, `per_level_mae_maps.png` is the generated numerical
source for the four error-map panels. The final paper composite at
`figures/GeometryAndFlowVisualization/CFD-FieldReconstruction-Flow.png`
also contains CFD/domain artwork and is manually composed; this repository does
not currently automate that final layout. The `.npz` file retains the underlying
maps if the panels need to be re-rendered.

## Supplementary analyses

These are useful comparisons, but **are not the publication configuration**:

* `field_reconstruction_baselayer_only.py` and
  `field_reconstruction_layer1_only.py`: single-level analyses.
* `field_reconstruction_multilayer.py`: 2D multilayer analysis.
* `hyperparameter_tuning_3dcnn.py`: hyperparameter-tuning analysis used to
  search candidates rather than report the final run.
* `run_3d_split_sensitivity.py` and `plot_3d_split_sensitivity.py`:
  split-sensitivity analysis and its summary plots.
