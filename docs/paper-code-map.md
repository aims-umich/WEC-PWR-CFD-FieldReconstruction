# Paper-to-code map

This guide maps the relevant sections, tables, and figures in
[`paper/main.tex`](../paper/main.tex) to the code currently available in this
repository. It distinguishes runnable repository code from external simulation
assets and implementations that have not yet been added.

> **Maintenance requirement:** update this map whenever a paper label, entry
> point, input/output contract, or availability status changes. In particular,
> replace the forthcoming entries when the remaining ML applications are added.

## Correspondence table

| Paper section or label | Topic | Implementation entry point | Supporting modules | Required inputs | Generated outputs | Current availability |
|---|---|---|---|---|---|---|
| `sec:models` — Reactor Modeling and Simulation | PWR geometry, porous-core representation, mesh setup, and assembly sensor/report preparation | STAR-CCM+ model setup is performed in the external simulation; repository automation is under [`java_macros/`](../java_macros/) | `ListPartIdentifiersMacro.java`, `DuplicateAndOffsetPlanesMacro.java`, `SetAbovePlanesOriginMacro.java`, `SetBelowPlanesOriginMacro.java`, `CreateMassFlowReports.java`, and `CreateMassFlowMonitorsPlot.java` | A licensed STAR-CCM+ installation and the project-specific STAR-CCM+ simulation/CAD model, including the expected part and report names | Modified STAR-CCM+ simulation objects: duplicated/offset planes, assembly mass-flow reports and monitors, and monitor plots | **Partial.** Supporting macros are included. The STAR-CCM+ simulation and CAD files are not present in this release; a public external download link is planned. |
| `sec:method` — Methodology | Transient runs, cold-leg swirl study automation, assembly-level mass-flow collection, and conversion of exported monitor data | [`java_macros/RunTestCaseMacroV5.java`](../java_macros/RunTestCaseMacroV5.java), [`java_macros/run_all.sh`](../java_macros/run_all.sh), then [`postprocessing/massflow_processing_pipeline.py`](../postprocessing/massflow_processing_pipeline.py) in processing mode | Remaining plane/report macros in `java_macros/`; `postprocessing/csvChecker.py` is a legacy standalone CSV/HDF5 utility | STAR-CCM+ simulation template and test-case CSV for simulation automation; exported `batch*.csv` files under `<case>/raw_csv/` for postprocessing | STAR-CCM+ run logs and exported mass-flow histories; combined/reordered CSVs in `<case>/intermediate/`; plane HDF5 files in `<case>/hdf5/`; mapping plots in `<case>/plots/`; optional videos in `<case>/video/` | **Partial.** Automation and preprocessing code are included. The simulation files and original simulation inputs are forthcoming through the planned external download; STAR-CCM+ remains separately licensed. |
| `sec:cfd_results` — CFD Simulation Results | Assembly flow maps and mesh-fidelity comparisons | [`postprocessing/massflow_processing_pipeline.py`](../postprocessing/massflow_processing_pipeline.py): normal processing or `--use-reordered` for maps; `--error-2f` for a comparison/reference pair; `--error-3f` for all three pairwise fidelity comparisons | Built-in 15×15/193-assembly mapping, time alignment, error metrics, and plotting functions in the same module | Exported batch CSVs or an existing `*_reordered.csv`; two case directories for `--error-2f`, or three case directories for `--error-3f`; optional `--core-mask-npy` | Combined/reordered CSVs, per-plane HDF5 datasets, mean-flow mapping plots, optional videos, and mesh-error figures under `<out_dir>/error_plots/` | **Included** for conversion and analysis. Reproduction of the underlying CFD solution requires the external STAR-CCM+ model and data, for which a public download link is planned. |
| `app:A` — Flow Error Results for all axial layers | Fine/medium/coarse maximum and time-averaged error maps across all nine layers | [`postprocessing/massflow_processing_pipeline.py`](../postprocessing/massflow_processing_pipeline.py) with `--error-3f` | `run_error_analysis_3f`, `compute_error_metrics`, `plot_error_base`, and `plot_error_grid_3x3` in the same module | Three case directories, each containing `intermediate/*_reordered.csv`; optional geometry mask and time-rounding settings | Three pairwise sets of base-layer and 3×3 all-layer error plots in `<out_dir>/error_plots/`, corresponding to the fine/medium, fine/coarse, and medium/coarse appendix comparisons | **Included** for analysis, subject to availability of the three reordered CFD datasets. |
| `sec:flow_recon` — Flow Field Reconstruction with CNN | Joint four-level 3D-CNN inpainting from a fixed partial-observation mask | [`experiments/field_reconstruction/field_reconstruction_multilayer_3DCNN.py`](../experiments/field_reconstruction/field_reconstruction_multilayer_3DCNN.py) | Reusable implementation in [`src/neup_inference_of_flow/field_reconstruction/`](../src/neup_inference_of_flow/field_reconstruction/): configuration, HDF5 I/O, preprocessing, masks, datasets, 3D model, training, metrics, pipelines, and plots | Four aligned plane HDF5 files (`plane_01_Base.h5` through `plane_04_Layer3.h5`) generated from the high-fidelity case; a required `--results-dir-name` | `config.json`, `dataset_paths.json`, `history.json`, `per_level_summary.json`, `per_level_arrays.npz`, `error_samples.npy`, `model_state_dict.pt`, `training_curves_3d.png`, `per_level_mae_maps.png`, and `per_level_error_histograms.png` under `experiments/field_reconstruction/results/<name>/` | **Included.** Input HDF5 data must exist at the paths configured by the entry point. |
| `tab:field-recon-results` — Plane-wise reconstruction performance | MAE, MAPE, and R² on intentionally hidden cells for each of four planes | The same 3D-CNN entry point as `sec:flow_recon` | `pipelines.run_multilevel_3d_experiment` and `metrics.compute_per_level_metrics` | The four aligned HDF5 inputs and the field-reconstruction experiment configuration | `per_level_summary.json` supplies the table-ready MAE, MAPE, R², and sample counts; `per_level_arrays.npz` retains the corresponding arrays | **Included.** The repository does not yet designate a single archival run directory as the authoritative source for the values typeset in the paper. |
| `fig:field-recon-flow` — Reconstruction workflow and plane-wise MAE visualization | Composite physical/workflow diagram plus spatial error maps | Numerical error maps are generated by the same 3D-CNN entry point; the paper consumes `figures/GeometryAndFlowVisualization/CFD-FieldReconstruction-Flow.png` | `metrics.compute_per_cell_error_maps`, `metrics.compute_per_level_metrics`, and `plots.plot_per_level_maps`; composition of the final paper graphic is not automated in the current code | Trained experiment outputs and the associated CFD/domain artwork | `per_level_mae_maps.png` provides the generated plane-wise MAE panels; the manually composed paper asset is `figures/GeometryAndFlowVisualization/CFD-FieldReconstruction-Flow.png` | **Partial.** Model-derived MAE maps are generated, but no repository script currently assembles the complete composite paper figure. |
| “One-Step Flow Field Prediction” | LSTM, ConvLSTM, and DeepONet one-step prediction comparison | None | None | Not specified in repository code | Not generated by this release | **Code forthcoming—not present in this release.** |
| `tab:ml_compNO` — ML model comparison | LSTM, ConvLSTM, and DeepONet metrics | None | None | Not specified in repository code | Not generated by this release | **Code forthcoming—not present in this release.** |
| `fig:first4_layer_ConvLSTM` — ConvLSTM spatial differences | First-four-layer prediction/error panels | None | None | Not specified in repository code | Not generated by this release | **Code forthcoming—not present in this release.** |
| `fig:summary_boxplot_ConvLSTM` — ConvLSTM summary boxplot | Cross-layer ConvLSTM error summary | None | None | Not specified in repository code | Not generated by this release | **Code forthcoming—not present in this release.** |

## Mesh-comparison CLI modes

Run commands from the repository root. Each case directory supplied to an error
mode must contain `intermediate/*_reordered.csv` (or the legacy
`intermediate/reordered.csv`).

Two-fidelity comparison:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --error-2f \
  --error-2f-comp-dir <comparison-case> \
  --error-2f-ref-dir <reference-case>
```

Three-fidelity/pairwise comparison used for the full mesh-sensitivity set:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --error-3f \
  --error-3f-case-a-dir <fine-case> \
  --error-3f-case-b-dir <medium-case> \
  --error-3f-case-c-dir <coarse-case>
```

The error modes write figures below `<out_dir>/error_plots/`. Use `--out-dir`
to make that destination explicit; otherwise the pipeline infers it from the
inputs or uses its dated fallback under `outputs/`.
