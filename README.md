# Inference of Flow Conditions from In-Core Detector Measurements for SMR Licensing

## Project and publication scope

This repository supports a broader NEUP project whose objective is to develop a
machine-learning-driven framework for inferring flow conditions and fuel bowing
from reactor detector measurements. The broader project combines CFD,
neutronics, detector data, and ML for reactor analysis and licensing research.

The **current publication reference is narrower than that project scope**. This
release contains the field-reconstruction ML workflow, reusable reconstruction
modules, CFD mass-flow postprocessing, and supporting STAR-CCM+ automation. See
the **[paper-to-code map](docs/paper-code-map.md)** for the entry points, inputs,
outputs, and availability associated with each relevant section of
`paper/main.tex`.

The LSTM, ConvLSTM, DeepONet, and other one-step flow-prediction workflows
described in the paper are **not present in this release**. They are planned for
a later addition; the paper-to-code map records these gaps explicitly and must
be updated when those implementations are added.

### Repository component status

| Component | Status | Notes |
|---|---|---|
| CFD model and simulation setup | Planned external download | The STAR-CCM+ simulation and CAD files are not included in this repository release. A public download link is planned; supporting automation macros are included here. |
| STAR-CCM+ data-collection automation | Partial | Macros and shell runners are included in `java_macros/`, but remain dependent on STAR-CCM+ and the project simulation setup. |
| CFD mass-flow conversion and mesh-comparison analysis | Included | CSV combination, HDF5 conversion, plots, videos, and two-/three-fidelity error modes are in `postprocessing/`. |
| CNN/3D-CNN field reconstruction | Included | Experiment entry points and reusable modules are included. |
| LSTM one-step prediction | Planned | Code forthcoming—not present in this release. |
| ConvLSTM one-step prediction | Planned | Code forthcoming—not present in this release. |
| DeepONet one-step prediction | Planned | Code forthcoming—not present in this release. |
| Neutronics, detector-inference, and fuel-bowing workflows from the broader project | Planned or external | Not part of the current publication code reference. |

### CAD and CFD model availability

The CAD geometry and STAR-CCM+ CFD model files are not stored in this Git
repository because of their size. They are planned for public release through
an external download link. That link will be added here when the archive is
available. Until then, the files should be considered **forthcoming**, not part
of the current repository release. The scripts and STAR-CCM+ macros already in
this repository remain available for inspecting and automating the associated
workflow.

## Problem statement
- **Challenge:** Reactor models lack accurate as-built conditions, leading to **unexplained power tilts** in **pressurized water reactors (PWRs)** during low-power physics tests.  
- **Hypothesis:** **Non-uniform inlet flow conditions** cause **fuel bowing**, altering neutron flux.  
- **Current Limitation:** Simulation tools struggle to capture these effects accurately.  

## Broader project approach
- **Use BEAVRS Benchmark:** Validate models using real-world **detector data from a PWR**.  
- **ML & CFD Modeling:** Train **ML models** to correlate **detector signals** with **fuel bowing and inlet flow variations** using **CASMO/SIMULATE, OpenMC, and STAR-CCM+ CFD**.  
- **Validation:** Compare model predictions with measured data to refine simulation accuracy.

![Project Workflow](figures/project_workflow.png)

## Broader project tasks
1. **Neutronic Modeling (MIT, UMich, NuScale):** Simulate **assembly gaps** and their impact on **detector signals**.  
2. **CFD Modeling (MIT, UMich, NuScale):** Develop **high-fidelity CFD models** to analyze core **inlet flow distributions** and **flow anomalies**.  
3. **ML Training (MIT, UMich, INL):** Use **CNNs & Bayesian ML** to infer flow conditions and fuel bowing from detector data.  
4. **Verification & Validation (MIT, UMich, INL):** Validate ML predictions using **BEAVRS and NuScale SMR models**.  
5. **Fuel Bowing Modeling (MIT, UMich):** Implement **3D bowed fuel models** in **OpenMC**.  
6. **Scalability & Open-Source Tools (All Partners):** Publish framework for broader **reactor licensing applications**.  

## Impact and relevance
- **Enhances SMR licensing** by improving reactor **safety and simulation accuracy**.  
- **Reduces uncertainty in core flow conditions, fuel bowing, and power tilts**.  
- **Improves prediction of key safety parameters** like **shutdown margin** and **critical heat flux**.  
- **Generalizable ML framework** applicable to **BWRs, SMRs, and advanced reactors**.  

This broader project leverages ML and simulations to infer reactor conditions,
improving predictive modeling, safety, and licensing for SMRs and advanced
reactors. The status table above, rather than the broader project task list,
defines what is actually distributed in this repository release.

---

## Repository layout

The repository uses top-level folders for each workflow area:

```text
.
├── experiments/
│   ├── field_reconstruction/       # Runnable reconstruction experiments and tuning utilities
│   └── one_step_prediction/        # Model-specific locations for future one-step experiments
├── src/
│   └── neup_inference_of_flow/
│       └── field_reconstruction/   # Reusable data, model, training, metrics, and plotting modules
├── postprocessing/                 # CFD CSV conversion, HDF5 generation, and fidelity analysis
├── java_macros/                    # STAR-CCM+ Java macros and shell runners
├── tests/                          # Automated tests for reusable Python functionality
├── paper/
│   └── main.tex                    # LaTeX source used to define the paper/code correspondence
├── docs/
│   └── paper-code-map.md           # Paper sections, entry points, inputs, outputs, and availability
├── POD_tutorials/                  # POD/SPOD exploratory notebooks and helper code
├── figures/                        # Figure assets and generated visualizations
├── presentations/                 # Non-code project presentation material
└── refs/                           # Non-code reference documents
```

Install the reusable package from the repository root before running an
experiment or importing it in another project:

```bash
python -m pip install -e .
python -c "import neup_inference_of_flow.field_reconstruction"
```

Field-reconstruction publication and supplementary commands are documented in
[`experiments/field_reconstruction/README.md`](experiments/field_reconstruction/README.md).
The model-specific one-step-prediction directories are reserved for their
respective runnable configurations; those workflows are not included in this
release yet.

### `postprocessing/` structure

`postprocessing/` contains both the CLI pipeline and case-study data/results. Current contents include:

```text
postprocessing/
├── massflow_processing_pipeline.py
├── csvChecker.py
├── nonporous_core_gaps/
│   └── linear_k-eps/
│       ├── FApitch_over12/
│       └── fine_mesh_02112026/
├── porous_full_core/
│   ├── cubic_k-eps/
│   │   ├── FApitch_over12/
│   │   └── FApitch_over14/
│   └── linear_k-eps/
│       ├── FApitch_over10/
│       └── FApitch_over12/
└── outputs/
    └── analysis_*/
```

---

## Mass flow processing pipeline usage (`postprocessing/massflow_processing_pipeline.py`)

This script combines detector/batch CSVs, reorders data by layer/fuel assembly, writes per-plane HDF5 files, generates mapping plots, and can also generate videos and fidelity error plots.

### 1) Required input case layout (processing mode)

For processing mode, input must look like:

```text
<parent>/FApitch_overX/
└── raw_csv/
    ├── batch1.csv
    ├── batch2.csv
    ├── batch3.csv
    └── batch4.csv
```

- `X` must be an integer (e.g., `FApitch_over12`).
- `--batches-dir` must point to `raw_csv/`.
- The script enforces the `FApitch_overX` case-directory naming convention.

### 2) Typical processing command

From repository root:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --batches-dir postprocessing/porous_full_core/linear_k-eps/FApitch_over12/raw_csv
```

Optional controls:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --batches-dir postprocessing/porous_full_core/linear_k-eps/FApitch_over12/raw_csv \
  --pattern "batch*.csv" \
  --t-min 10 --t-max 20
```

### 3) Where outputs are saved

#### A) Processing mode (`--batches-dir .../FApitch_overX/raw_csv`)
Outputs are written to the case directory that contains `raw_csv/`:

```text
.../FApitch_overX/
├── intermediate/
│   ├── FApitch_overX_combined.csv
│   └── FApitch_overX_reordered.csv
├── hdf5/
│   └── plane_XX_<LayerTag>.h5
├── plots/
│   ├── mapping_avg_shared_colorbar_setYY.png
│   └── mapping_avg_per_layer_colorbars_setYY.png
└── video/                  # only when --video is enabled
    ├── video_*_shared.mp4
    └── video_*_per_subplot.mp4
```

#### B) Analysis from existing reordered CSV (`--use-reordered`)
- If the input path is `.../intermediate/*_reordered.csv`, outputs are saved under that case directory.
- Otherwise, outputs are written to a dated fallback folder: `outputs/analysis_<MMDDYYYY>/`.

#### C) Error analysis modes (`--error-2f` / `--error-3f`)
- Error plots are written under `<out_dir>/error_plots/`.
- By default, `out_dir` is inferred from input paths when possible; otherwise it falls back to `outputs/analysis_<MMDDYYYY>/`.

### 4) Video generation examples

Generate videos from an existing reordered CSV:

```bash
python postprocessing/massflow_processing_pipeline.py \
  --use-reordered postprocessing/porous_full_core/linear_k-eps/FApitch_over12/intermediate/FApitch_over12_reordered.csv \
  --video --video-fps 60 --video-every 20
```

Useful options:
- `--video-vmin`, `--video-vmax`
- `--video-max-frames`
- `--video-keep-frames`

### 5) 2-fidelity error analysis example

```bash
python postprocessing/massflow_processing_pipeline.py \
  --error-2f \
  --error-2f-comp-dir postprocessing/porous_full_core/linear_k-eps/FApitch_over12 \
  --error-2f-ref-dir  postprocessing/porous_full_core/linear_k-eps/FApitch_over10
```

### 6) 3-fidelity error analysis example

```bash
python postprocessing/massflow_processing_pipeline.py \
  --error-3f \
  --error-3f-case-a-dir postprocessing/porous_full_core/cubic_k-eps/FApitch_over12 \
  --error-3f-case-b-dir postprocessing/porous_full_core/linear_k-eps/FApitch_over10 \
  --error-3f-case-c-dir postprocessing/porous_full_core/linear_k-eps/FApitch_over12
```

Shared error options:
- `--core-mask-npy <path/to/mask.npy>`
- `--error-time-round <N>`
- `--error-asymmetric`

### 7) Quick troubleshooting

- Case directory must be named exactly `FApitch_overX`.
- Processing input must be `.../FApitch_overX/raw_csv`.
- Each error-analysis case must have `intermediate/*_reordered.csv` (or `reordered.csv`).
- For video mode, OpenCV (`cv2`) must be installed.
