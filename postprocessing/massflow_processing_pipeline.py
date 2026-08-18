#!/usr/bin/env python3
"""massflow_processing_pipeline.py

Single CLI pipeline refactoring the 3-notebook workflow:

1) RemoveTimeColsAndCombineCSVBatches
2) ReorderCSVGroupByLayer
3) processMFcsv (+ required mapping plot checks)

Pipeline:
  batch CSVs (from <...>/FApitch_overX/raw_csv) -> FApitch_overX_combined.csv -> FApitch_overX_reordered.csv
  -> hdf5 planes + composite mapping plots

Outputs are written under the case directory (same parent as raw_csv):
  <...>/FApitch_overX/intermediate
  <...>/FApitch_overX/hdf5
  <...>/FApitch_overX/plots

Usage:
  # Provide a directory that contains batch CSVs
  python massflow_processing_pipeline.py --batches-dir ./batches

  # Custom filename pattern
  python massflow_processing_pipeline.py --batches-dir ./batches --pattern "*FApitch_over12*.csv"

  # Explicit batch list (order preserved)
  python massflow_processing_pipeline.py --batches-dir ./batches

  # Restrict the time window used from the batches (based on the FIRST column of each batch)
  python massflow_processing_pipeline.py --batches-dir ./batches --t-min 10 --t-max 20

Notes / invariants (hard-fail):
- The FIRST column of every batch CSV is treated as the time column for windowing.
- Batch time vectors must match exactly across all batches.
- Every sensor column must parse BOTH:
    - FA### (FA001..FA193)
    - a layer tag: Base or LayerN (also accepts L4, Layer 4)
- Every layer must have EXACTLY 193 columns covering FA001..FA193.
- Combined/reordered outputs must not contain duplicate column names.
- Refuses to overwrite an existing output directory unless --overwrite is provided.
- Mapping plot check is always generated as 2 composite 3x3 grids per set of 9 planes:
    1) Shared (single) colorbar across all subplots
    2) One colorbar per subplot (9 total)

"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Parsing + mapping helpers
# =============================================================================

_FA_RE = re.compile(r"FA(\d{3})", re.IGNORECASE)
_LAYER_RE = re.compile(r"(base)|(layer\s*([0-9]+))|\bL([0-9]+)\b", re.IGNORECASE)
_CASE_DIR_RE = re.compile(r"^FApitch_over(\d+)$")
VIDEO_BASE_LAYER_VMIN = 65.0
VIDEO_BASE_LAYER_VMAX = 95.0

CUSTOM_32_COLORS = [
    "#1c1d8a", "#1a2988", "#173386", "#153b84",
    "#124282", "#164e8f", "#1b5a9e", "#2064ac",
    "#246db8", "#2776c4", "#408bc8", "#519ecb",
    "#60aecf", "#75bec7", "#a3d27f", "#dbec62",
    "#fcf957", "#fded55", "#fde052", "#fdd24f",
    "#fec34c", "#feb349", "#fea246", "#ff8e43",
    "#fe7c3f", "#fb703a", "#f76334", "#f4542e",
    "#f14127", "#d83820", "#b4331a", "#882e10",
]


def get_massflow_cmap(use_custom: bool):
    if use_custom:
        from matplotlib.colors import ListedColormap

        return ListedColormap(CUSTOM_32_COLORS, name="custom_32")
    return "rainbow"


def extract_ca_id(colname: str) -> Optional[int]:
    """Extract FA id as an integer (1..193) from a column name containing FA###."""
    m = _FA_RE.search(colname)
    if not m:
        return None
    return int(m.group(1))


def extract_layer_tag(colname: str) -> Optional[str]:
    """Extract standardized layer tag: 'Base' or 'LayerN'. Returns None if not found."""
    m = _LAYER_RE.search(colname)
    if not m:
        return None
    if m.group(1):
        return "Base"
    num = m.group(3) or m.group(4)
    if num is None:
        return None
    return f"Layer{int(num)}"


def _layer_sort_key(tag: str) -> Tuple[int, int]:
    """Sort Base first, then Layer1..N."""
    if tag.lower() == "base":
        return (0, 0)
    m = re.match(r"layer(\d+)$", tag.lower())
    if m:
        return (1, int(m.group(1)))
    return (2, 9999)


def core_mask_15x15() -> np.ndarray:
    """Create a 15x15 boolean mask with 193 active cells (core assemblies)."""
    ring_counts = [7, 11, 13, 13]
    offsets = [4, 2, 1, 1]
    counts_full = ring_counts + ring_counts[::-1]
    offs_full = offsets + offsets[::-1]

    rows: List[np.ndarray] = []
    for cnt, off in zip(counts_full[:4], offs_full[:4]):
        r = np.zeros(15, dtype=bool)
        r[off : off + cnt] = True
        rows.append(r)
    rows += [np.ones(15, dtype=bool) for _ in range(7)]
    for cnt, off in zip(counts_full[4:], offs_full[4:]):
        r = np.zeros(15, dtype=bool)
        r[off : off + cnt] = True
        rows.append(r)

    mask = np.vstack(rows)
    if mask.shape != (15, 15) or int(mask.sum()) != 193:
        raise RuntimeError(
            f"core_mask_15x15 produced shape={mask.shape}, sum={mask.sum()} (expected 15x15, 193)"
        )
    return mask


def reshape_193_to_15x15(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-major fill across the 15×15 core mask (top→bottom, left→right)."""
    if vals.size != int(mask.sum()):
        raise ValueError(f"Expected {int(mask.sum())} sensors, got {vals.size}")
    grid = np.zeros((15, 15), dtype=float)
    grid[mask] = vals.astype(float)
    return grid


# =============================================================================
# Utility + validation
# =============================================================================


def _die(msg: str) -> None:
    raise SystemExit(f"\n[ERROR] {msg}\n")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _assert_no_duplicate_columns(cols: List[str], context: str) -> None:
    dupes = pd.Index(cols).duplicated()
    if dupes.any():
        dupe_names = pd.Index(cols)[dupes].unique().tolist()
        _die(
            f"Duplicate columns detected in {context}: {dupe_names[:20]}"
            + (" ..." if len(dupe_names) > 20 else "")
        )


def _validate_time_alignment(master_time: np.ndarray, other_time: np.ndarray, path: Path) -> None:
    if master_time.shape != other_time.shape:
        _die(f"Time vector length mismatch for {path.name}: {other_time.shape} vs master {master_time.shape}")
    if not np.allclose(master_time, other_time, rtol=0, atol=0):
        diffs = np.where(master_time != other_time)[0]
        i0 = int(diffs[0])
        _die(
            f"Time vectors do not match exactly for {path.name}. "
            f"First mismatch at index {i0}: master={master_time[i0]} vs other={other_time[i0]}"
        )


def _natural_sort_key(name: str) -> List[object]:
    """Sort keys like batch2 < batch10."""
    parts = re.split(r"(\d+)", name)
    out: List[object] = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def discover_batch_files(batches_dir: Path, pattern: str) -> List[Path]:
    if not batches_dir.exists():
        _die(f"Batches directory not found: {batches_dir}")
    paths = sorted(batches_dir.glob(pattern), key=lambda p: _natural_sort_key(p.name))
    if not paths:
        _die(f"No batch CSVs found in '{batches_dir}' matching pattern '{pattern}'.")
    print("[info] Discovered batch files (in concat order):")
    for p in paths:
        print(f"  - {p}")
    return paths


# =============================================================================
# Step 1: Combine batches (with optional time-window selection)
# =============================================================================


def combine_batches(
    batch_paths: List[Path],
    out_csv: Path,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> Path:
    """Combine multiple batch CSV files by concatenating sensor columns side-by-side.

    The FIRST column of each batch is treated as the time column.

    If t_min and/or t_max are provided, rows are filtered to the inclusive window:
      t_min <= t <= t_max

    Time vectors across batches must match EXACTLY.
    """
    if not batch_paths:
        _die("No batch paths provided.")

    for p in batch_paths:
        if not p.exists():
            _die(f"Missing batch CSV: {p}")

    # Read first batch to establish master time + row mask
    df0 = pd.read_csv(batch_paths[0])
    if df0.shape[1] < 2:
        _die(f"Batch CSV has <2 columns (time + sensors expected): {batch_paths[0]}")

    full_master_time = df0.iloc[:, 0].to_numpy()

    mask = np.ones_like(full_master_time, dtype=bool)
    if t_min is not None:
        mask &= full_master_time >= float(t_min)
    if t_max is not None:
        mask &= full_master_time <= float(t_max)

    if mask.sum() == 0:
        _die(
            "Time-window selection produced 0 rows. "
            f"Requested t_min={t_min}, t_max={t_max}; "
            f"available range is [{full_master_time.min()}, {full_master_time.max()}]."
        )

    master_time = full_master_time[mask]

    if t_min is None and t_max is None:
        print(f"[info] Using full time duration: rows={len(master_time)}")
    else:
        print(
            f"[info] Using time window: t_min={t_min}, t_max={t_max} -> rows={len(master_time)}"
        )

    batch_dfs: List[pd.DataFrame] = []

    for i, p in enumerate(batch_paths):
        df = pd.read_csv(p)
        time_vals = df.iloc[:, 0].to_numpy()

        if i > 0:
            _validate_time_alignment(full_master_time, time_vals, p)

        # Apply the row mask (all batches share identical time indexing)
        df = df.iloc[mask, :]

        # Drop first column (time)
        df_sensors = df.iloc[:, 1:]

        # Defensive: drop any stray time-like columns remaining
        drop_cols = []
        for c in df_sensors.columns:
            cl = str(c).strip().lower()
            if cl == "time" or "physical time (s)" in cl:
                drop_cols.append(c)
        if drop_cols:
            df_sensors = df_sensors.drop(columns=drop_cols)

        batch_dfs.append(df_sensors)

    combined = pd.concat(batch_dfs, axis=1)
    combined.insert(0, "Physical Time (s)", master_time)

    _assert_no_duplicate_columns(list(combined.columns), context=f"combined CSV ({out_csv.name})")

    case_label = infer_case_label_from_path(batch_paths[0])
    target_csv = out_csv
    if out_csv.name.lower() == "combined.csv":
        target_csv = out_csv.with_name(preferred_csv_name("combined", case_label))

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target_csv, index=False)

    print(f"[ok] Combined {len(batch_paths)} batches -> {target_csv}  shape={combined.shape}")
    return target_csv


# =============================================================================
# Step 2: Reorder by layer and FA
# =============================================================================


def reorder_csv_group_by_layer(in_csv: Path, out_csv: Path) -> Path:
    if not in_csv.exists():
        _die(f"Missing input CSV for reorder step: {in_csv}")

    df = pd.read_csv(in_csv)

    # Identify time column
    if "Physical Time (s)" in df.columns:
        time_col = "Physical Time (s)"
    elif "Time" in df.columns:
        time_col = "Time"
    else:
        time_col = df.columns[0]
        _warn(f"No explicit time column found. Using first column as time: '{time_col}'")

    sensor_cols = [c for c in df.columns if c != time_col]

    parsed: Dict[str, Tuple[str, int]] = {}  # col -> (layer_tag, fa_id)
    unmatched: List[str] = []

    for c in sensor_cols:
        s = str(c)
        fa = extract_ca_id(s)
        layer = extract_layer_tag(s)
        if fa is None or layer is None:
            unmatched.append(s)
        else:
            parsed[s] = (layer, fa)

    if unmatched:
        preview = "\n  - ".join(unmatched[:20])
        _die(
            f"{len(unmatched)} columns could not be parsed for BOTH FA### and layer tag.\n"
            f"First examples:\n  - {preview}\n"
            "Expected patterns like '...FA001_Base...' or '...FA093_Layer4...' or '..._L4...'."
        )

    by_layer: Dict[str, List[Tuple[int, str]]] = {}
    for col, (layer, fa) in parsed.items():
        by_layer.setdefault(layer, []).append((fa, col))

    expected_fas = set(range(1, 194))
    layer_counts = {layer: len(items) for layer, items in by_layer.items()}
    print("[info] Parsed layers:", {k: layer_counts[k] for k in sorted(layer_counts, key=_layer_sort_key)})

    for layer, items in by_layer.items():
        fas = [fa for fa, _ in items]
        fas_set = set(fas)

        if len(items) != 193:
            missing = sorted(expected_fas - fas_set)
            extra = sorted(fas_set - expected_fas)
            _die(
                f"Layer '{layer}' has {len(items)} columns (expected 193).\n"
                f"Missing FAs: {missing[:25]}{' ...' if len(missing) > 25 else ''}\n"
                f"Extra FAs: {extra[:25]}{' ...' if len(extra) > 25 else ''}"
            )

        if fas_set != expected_fas:
            _die(f"Layer '{layer}' FA set mismatch vs 1..193.")

    ordered_layers = sorted(by_layer.keys(), key=_layer_sort_key)

    ordered_sensor_cols: List[str] = []
    for layer in ordered_layers:
        items = sorted(by_layer[layer], key=lambda t: t[0])
        ordered_sensor_cols.extend([col for _, col in items])

    out_aols = [time_col] + ordered_sensor_cols

    # If any columns were renamed by pandas (unlikely) we need to match original columns
    out_df = df.loc[:, out_aols]

    _assert_no_duplicate_columns(list(out_df.columns), context=f"reordered CSV ({out_csv.name})")

    case_label = infer_case_label_from_path(in_csv)
    target_csv = out_csv
    if out_csv.name.lower() == "reordered.csv":
        target_csv = out_csv.with_name(preferred_csv_name("reordered", case_label))

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_csv, index=False)

    print(f"[ok] Reordered -> {target_csv}  shape={out_df.shape}")
    return target_csv


# =============================================================================
# Step 3: CSV -> per-plane HDF5 + composite mapping plots
# =============================================================================


def _save_mapping_grids_3x3(
    avg_maps: List[np.ndarray],
    tags: List[str],
    plots_dir: Path,
    set_index: int,
    case_name: Optional[str] = None,
    use_custom_cmap: bool = False,
) -> None:
    """Save two 3x3 mapping grid figures for a set of up to 9 planes.

    Figure A: shared single colorbar across all subplots (common scale)
    Figure B: one colorbar per subplot (9 total)
    """

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Always a 3x3 layout; if fewer than 9, leave remaining axes blank.
    n = len(avg_maps)
    if n == 0:
        return
    if n < 9:
        _warn(f"Only {n} planes available for mapping grid set {set_index:02d}. Remaining subplots will be empty.")

    # Common color limits (ignore NaNs)
    vmin = float(np.nanmin([np.nanmin(m) for m in avg_maps]))
    vmax = float(np.nanmax([np.nanmax(m) for m in avg_maps]))

    cmap = get_massflow_cmap(use_custom_cmap)

    # --------------------------
    # Figure 1: shared colorbar
    # --------------------------
    fig1, axes1 = plt.subplots(3, 3, figsize=(12, 12))
    axes1_flat = axes1.ravel()

    last_im = None
    for i in range(9):
        ax = axes1_flat[i]
        if i < n:
            im = ax.imshow(avg_maps[i], interpolation="none", vmin=vmin, vmax=vmax, cmap=cmap)
            last_im = im
            ax.set_title(f"Plane {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    if last_im is not None:
        cbar = fig1.colorbar(last_im, ax=axes1_flat.tolist(), shrink=0.8)
        cbar.set_label("Mass flow rate (kg/s)")

    fig1.suptitle(_with_case_prefix(f"Time-averaged mass flow mapping (shared colorbar) — set {set_index:02d}", case_name))
    #fig1.tight_layout(rect=[0, 0, 1, 0.95])

    out1 = plots_dir / f"mapping_avg_shared_colorbar_set{set_index:02d}.png"
    fig1.savefig(out1, dpi=200)
    plt.close(fig1)
    print(f"[ok] Saved mapping grid (shared colorbar) -> {out1}")

    # --------------------------
    # Figure 2: per-subplot colorbars
    # --------------------------
    fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))
    axes2_flat = axes2.ravel()

    for i in range(9):
        ax = axes2_flat[i]
        if i < n:
            im = ax.imshow(avg_maps[i], interpolation="none", cmap=cmap)
            ax.set_title(f"Plane {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        else:
            ax.axis("off")

    fig2.suptitle(_with_case_prefix(f"Time-averaged mass flow mapping (per-layer colorbars) — set {set_index:02d}", case_name))
    #fig2.tight_layout(rect=[0, 0, 1, 0.95])

    out2 = plots_dir / f"mapping_avg_per_layer_colorbars_set{set_index:02d}.png"
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"[ok] Saved mapping grid (per-layer colorbars) -> {out2}")


def csv_to_hdf5_planes_and_plots(
    in_csv: Path,
    hdf5_dir: Path,
    plots_dir: Path,
    case_name: Optional[str] = None,
    use_custom_cmap: bool = False,
) -> List[Path]:
    if not in_csv.exists():
        _die(f"Missing input CSV for HDF5 step: {in_csv}")

    df = pd.read_csv(in_csv)

    time_col = df.columns[0]
    times = df[time_col].to_numpy()
    sensors = df.iloc[:, 1:].to_numpy()
    headers = list(df.columns[1:])

    T = sensors.shape[0]
    N = sensors.shape[1]

    if N % 193 != 0:
        _die(f"Sensor column count N={N} is not a multiple of 193. (Did reorder step succeed?)")

    P = N // 193

    hdf5_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mask = core_mask_15x15()
    h5_paths: List[Path] = []

    avg_maps_all: List[np.ndarray] = []
    tags_all: List[str] = []

    for p in range(P):
        col_lo = p * 193
        col_hi = (p + 1) * 193
        plane_slice = sensors[:, col_lo:col_hi]
        block_cols = headers[col_lo:col_hi]

        tags = [extract_layer_tag(str(c)) for c in block_cols]
        unique = {t for t in tags if t is not None}
        if not unique:
            _die(
                f"Could not infer a layer tag (Base/Layer#) for plane block {p+1}. "
                f"First col: {block_cols[0]}"
            )
        counts = {t: tags.count(t) for t in unique}
        tag = max(counts, key=counts.get)
        if len(unique) > 1:
            _warn(f"Mixed layer tags within plane block {p+1}: {counts}. Using '{tag}'.")

        safe_tag = tag.replace(" ", "")
        h5_path = hdf5_dir / f"plane_{p+1:02d}_{safe_tag}.h5"

        arr3d = np.zeros((T, 15, 15), dtype=float)
        for t in range(T):
            arr3d[t] = reshape_193_to_15x15(plane_slice[t], mask)

        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("time", data=times)
            h5.create_dataset("data", data=arr3d)

        h5_paths.append(h5_path)
        print(f"[ok] Saved {h5_path}  time={times.shape}  data={arr3d.shape}")

        avg = np.nanmean(arr3d, axis=0)
        avg_plot = avg.copy()
        avg_plot[~mask] = np.nan

        avg_maps_all.append(avg_plot)
        tags_all.append(safe_tag)

    # Composite mapping plots in 3x3 grids (two figures per set of 9 planes)
    if P == 0:
        _warn("No planes detected; skipping mapping grid plots.")
        return h5_paths

    for set_idx, start in enumerate(range(0, P, 9), start=1):
        end = min(start + 9, P)
        _save_mapping_grids_3x3(
            avg_maps=avg_maps_all[start:end],
            tags=tags_all[start:end],
            plots_dir=plots_dir,
            set_index=set_idx,
            case_name=case_name,
            use_custom_cmap=use_custom_cmap,
        )

    return h5_paths


# =============================================================================
# Step 2.5 (optional): Video generation from reordered.csv
# =============================================================================

def generate_video_from_reordered_csv(
    reordered_csv: Path,
    video_dir: Path,
    case_name: Optional[str] = None,
    fps: int = 15,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    max_frames: Optional[int] = None,
    *,
    keep_frames: bool = False,
    every_n: int = 20,
    colorbar_mode: str = "shared",
    use_custom_cmap: bool = False,
) -> Path:
    """
    Generate a 3x3 heatmap video (Base + Layer1..Layer8) from the reordered CSV.

    Sampling:
      - Only every Nth timestep is rendered (every_n). Example: T=100, every_n=5 -> 20 frames.

    Frame saving:
      - If keep_frames=True, PNGs are written to: video_dir/frames_<stem>/
      - If keep_frames=False (default), frames are streamed directly into the MP4 (no per-frame files).

    Colorbars:
      - colorbar_mode="shared": one shared colorbar for the full 3x3 grid (common scale).
      - colorbar_mode="per_subplot": one colorbar per subplot (9 total). If vmin/vmax are not
        provided, each layer uses its own fixed scale (min/max over all timesteps) to avoid flicker.

    The CSV must contain:
      - a time column named one of: Time, time, Physical Time (s), Physical Time, PhysicalTime(s)
      - 9 layers: Base + Layer1..Layer8
      - for each layer: 193 monitor columns matching: MON_MFR_FA###_(Base|Layer[1-8])
        (the match can occur anywhere in the column name; extra text like units is ok)
    """
    if not reordered_csv.exists():
        _die(f"Missing reordered CSV for video step: {reordered_csv}")

    if every_n is None or int(every_n) < 1:
        _die(f"--video-every must be >= 1 (got {every_n}).")
    every_n = int(every_n)

    # Import cv2 lazily so the rest of the pipeline works without it.
    try:
        import cv2  # type: ignore
    except Exception as e:
        _die(
            "OpenCV (cv2) is required for --video but was not found.\n"
            "In conda, the package name is 'opencv' (module name is 'cv2'). Try ONE of:\n"
            "  - conda install -c conda-forge opencv\n"
            "  - python -m pip install opencv-python\n"
            f"Import error: {e}"
        )

    video_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(reordered_csv)

    # ---- Time column detection ----
    time_candidates = ["Time", "time", "Physical Time (s)", "Physical Time", "PhysicalTime(s)"]
    time_col_name = next((c for c in time_candidates if c in df.columns), None)
    if time_col_name is None:
        _die(
            f"No time column found in {reordered_csv}.\n"
            f"Tried: {', '.join(time_candidates)}"
        )

    time_vals = df[time_col_name].to_numpy()

    # ---- Identify monitor columns per layer ----
    layer_names = ["Base"] + [f"Layer{i}" for i in range(1, 9)]
    layer_col_map: Dict[str, List[Tuple[int, str]]] = {layer: [] for layer in layer_names}

    pat = re.compile(r"MON_MFR_FA(\d{3})_(Base|Layer[1-8])", re.IGNORECASE)

    for col in df.columns:
        m = pat.search(str(col))
        if not m:
            continue

        fa_num = int(m.group(1))
        lyr = m.group(2)
        if lyr.lower() == "base":
            lyr_key = "Base"
        else:
            # Layer1..Layer8
            lyr_key = f"Layer{int(lyr[-1])}"

        layer_col_map[lyr_key].append((fa_num, str(col)))

    # Sort and validate counts
    for lyr in layer_names:
        items = sorted(layer_col_map[lyr], key=lambda x: x[0])
        cols_only = [c for _, c in items]
        if len(cols_only) != 193:
            preview = "\n  - ".join(cols_only[:10])
            _die(
                f"Video step: Layer '{lyr}' has {len(cols_only)} matched columns (expected 193).\n"
                "This usually means the column naming doesn't include the expected substring:\n"
                "  MON_MFR_FA###_(Base|Layer[1-8])\n"
                f"First matches:\n  - {preview}"
            )
        layer_col_map[lyr] = [(fa, col) for fa, col in items]  # keep sorted for later

    # ---- Core geometry layout (193 positions) ----
    layout_rows = [7, 11, 13, 13] + [15] * 7 + [13, 13, 11, 7]
    grid_h = len(layout_rows)
    grid_w = 15

    core_layout = np.full((grid_h, grid_w), -1, dtype=int)
    fa_counter = 0
    for r, row_len in enumerate(layout_rows):
        start_c = (grid_w - row_len) // 2
        for j in range(row_len):
            core_layout[r, start_c + j] = fa_counter
            fa_counter += 1

    if fa_counter != 193:
        _die(f"Video step: core layout assigned {fa_counter} positions; expected 193.")

    idx_to_rc: List[Tuple[int, int]] = [(-1, -1)] * 193
    for r in range(grid_h):
        for c in range(grid_w):
            idx = int(core_layout[r, c])
            if idx >= 0:
                idx_to_rc[idx] = (r, c)

    if any(r < 0 for r, _ in idx_to_rc):
        missing = [i for i, (r, _) in enumerate(idx_to_rc) if r < 0]
        _die(f"Video step: missing layout positions for indices: {missing[:10]} ...")

    # ---- Output paths ----
    stem = reordered_csv.stem  # expected: "reordered"

    if colorbar_mode not in ("shared", "per_subplot"):
        _die(f"Video step: invalid colorbar_mode='{colorbar_mode}'. Use 'shared' or 'per_subplot'.")

    suffix = "" if colorbar_mode == "shared" else "_per_subplot_cbar"
    out_mp4 = video_dir / f"video_{stem}{suffix}.mp4"

    frame_dir: Optional[Path] = None
    if keep_frames:
        frame_dir = video_dir / f"frames_{stem}{suffix}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        # Clear old PNGs so reruns are clean
        for f in frame_dir.glob("*.png"):
            try:
                f.unlink()
            except Exception:
                pass

    # Build per-layer views (columns are already sorted by FA)
    layer_cols_only = {lyr: [col for _, col in layer_col_map[lyr]] for lyr in layer_names}
    layer_dfs = [df[layer_cols_only[lyr]] for lyr in layer_names]

    # Determine color scale
    if vmin is None or vmax is None:
        data_min = min(float(ld.values.min()) for ld in layer_dfs)
        data_max = max(float(ld.values.max()) for ld in layer_dfs)
        vmin2 = data_min if vmin is None else float(vmin)
        vmax2 = data_max if vmax is None else float(vmax)
    else:
        vmin2, vmax2 = float(vmin), float(vmax)

    # Per-layer fixed limits (used for colorbar_mode="per_subplot" to avoid flicker)
    layer_vmins = [float(ld.values.min()) for ld in layer_dfs]
    layer_vmaxs = [float(ld.values.max()) for ld in layer_dfs]
    layer_vmin2 = [float(vmin) if vmin is not None else mv for mv in layer_vmins]
    layer_vmax2 = [float(vmax) if vmax is not None else mx for mx in layer_vmaxs]


    # ---- Decide which timesteps to render ----
    timestep_indices = list(range(0, len(time_vals), every_n))
    if max_frames is not None:
        timestep_indices = timestep_indices[: int(max_frames)]

    if not timestep_indices:
        _die("Video step: 0 timesteps selected for rendering (check --video-every / --video-max-frames).")

    print(
        f"[info] Video: mode={colorbar_mode}  timesteps={len(time_vals)}  stride={every_n}  "
        f"render_frames={len(timestep_indices)}  keep_frames={keep_frames}  fps={fps}"
    )

    # ---- Video writer helpers ----
    def _open_writer(size_wh: Tuple[int, int], fourcc_str: str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), size_wh)
        return vw if vw.isOpened() else None

    writer = None
    cmap = get_massflow_cmap(use_custom_cmap)

    # ---- Frame generation + streaming write ----
    for frame_idx, t in enumerate(timestep_indices):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        last_im = None

        for i, layer_df in enumerate(layer_dfs):
            ax = axes[i // 3, i % 3]
            values = layer_df.iloc[t].to_numpy(dtype=float)

            heatmap = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
            for idx, val in enumerate(values):
                r, c = idx_to_rc[idx]
                heatmap[r, c] = val

            # Force Base/first-plane subplot color limits in video to 65..95.
            if i == 0:
                subplot_vmin, subplot_vmax = VIDEO_BASE_LAYER_VMIN, VIDEO_BASE_LAYER_VMAX
            elif colorbar_mode == "shared":
                subplot_vmin, subplot_vmax = vmin2, vmax2
            else:
                subplot_vmin, subplot_vmax = layer_vmin2[i], layer_vmax2[i]

            im = ax.imshow(heatmap, cmap=cmap, vmin=subplot_vmin, vmax=subplot_vmax)
            if colorbar_mode == "per_subplot":
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            last_im = im
            ax.set_title(layer_names[i])
            ax.axis("off")

        fig.suptitle(_with_case_prefix(f"Time = {float(time_vals[t]):.4f} s", case_name), fontsize=18)
        if colorbar_mode == "shared" and last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6)

        # Render to an in-memory image (no disk required)
        # NOTE: Different matplotlib backends expose different canvas buffer APIs.
        # Prefer RGBA buffer if available; otherwise fall back to RGB/ARGB.
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        if hasattr(fig.canvas, "buffer_rgba"):
            # (h, w, 4) uint8 RGBA
            rgba = np.asarray(fig.canvas.buffer_rgba())
            # Some backends return a view; make it contiguous for OpenCV
            if not rgba.flags["C_CONTIGUOUS"]:
                rgba = np.ascontiguousarray(rgba)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        elif hasattr(fig.canvas, "tostring_rgb"):
            # (h, w, 3) uint8 RGB
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif hasattr(fig.canvas, "tostring_argb"):
            # (h, w, 4) uint8 ARGB -> convert to RGBA -> BGR
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
            rgba = argb[..., [1, 2, 3, 0]]
            if not rgba.flags["C_CONTIGUOUS"]:
                rgba = np.ascontiguousarray(rgba)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        else:
            plt.close(fig)
            _die(
                "Video step: cannot extract pixel buffer from matplotlib canvas. "
                "Try installing a non-interactive backend or force Agg."
            )

        # Initialize writer on first rendered frame (needs size)
        if writer is None:
            writer = _open_writer((w, h), "mp4v") or _open_writer((w, h), "avc1")
            if writer is None:
                plt.close(fig)
                _die(
                    "Video step: failed to open an MP4 VideoWriter.\n"
                    "On some Windows installs, this is a codec/ffmpeg issue.\n"
                    "If you're using conda-forge, try:\n"
                    "  conda install -c conda-forge ffmpeg\n"
                )

        writer.write(bgr)

        if keep_frames and frame_dir is not None:
            frame_path = frame_dir / f"frame_{frame_idx:04d}.png"
            fig.savefig(frame_path, dpi=150)  # do NOT bbox_inches="tight" (keeps constant size)

        plt.close(fig)

    if writer is None:
        _die("Video step: writer was never initialized (unexpected).")

    writer.release()
    print(f"[ok] Video generated -> {out_mp4}")

    if keep_frames and frame_dir is not None:
        print(f"[ok] Saved frames -> {frame_dir}")

    return out_mp4


# =============================================================================
# Step 4 (optional): Error analysis (2-fidelity / 3-fidelity)
# =============================================================================

_PCT_EPS = 1e-12


def load_massflow_csv_auto_sep(path: Path) -> pd.DataFrame:
    """Load a CSV that may be comma- or semicolon-delimited.

    Strategy:
      1) try ';'  (common for some STAR exports)
      2) if it collapses to 1 column, retry ','
    """
    if not path.exists():
        _die(f"Missing CSV: {path}")

    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    if df.shape[1] == 1:
        # Last resort: let pandas sniff delimiter (slower)
        df = pd.read_csv(path, sep=None, engine="python")
    return df


def detect_time_column(df: pd.DataFrame) -> str:
    """Return the time column name used by analysis steps."""
    candidates = ["Physical Time (s)", "Time", "time", "Physical Time", "PhysicalTime(s)"]
    for c in candidates:
        if c in df.columns:
            return c
    return str(df.columns[0])


def standardize_to_layers_193(
    df: pd.DataFrame,
    *,
    require_layers: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Parse columns into (time, data, layer_names).

    Returns:
      time: (T,)
      data: (T, L, 193) with FA order 1..193 for each layer
      layer_names: length L

    Column parsing uses extract_ca_id() + extract_layer_tag() from the pipeline.
    """
    time_col = detect_time_column(df)
    time_vals = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    sensor_cols = [c for c in df.columns if str(c) != time_col]

    # Build mapping: layer -> fa -> column name
    layer_map: Dict[str, Dict[int, str]] = {}
    bad_cols: List[str] = []

    for c in sensor_cols:
        s = str(c)
        fa = extract_ca_id(s)
        layer = extract_layer_tag(s)
        if fa is None or layer is None:
            bad_cols.append(s)
            continue
        if require_layers is not None and layer not in require_layers:
            continue
        layer_map.setdefault(layer, {})
        # If duplicates exist for the same (layer, fa), keep the first and warn.
        if fa in layer_map[layer]:
            _warn(f"Duplicate column for {layer} FA{fa:03d} in CSV; keeping first: '{layer_map[layer][fa]}' and ignoring '{s}'")
            continue
        layer_map[layer][fa] = s

    if require_layers is not None:
        missing_layers = [l for l in require_layers if l not in layer_map]
        if missing_layers:
            _die(f"Missing required layers in CSV: {missing_layers}")

    # Decide layer order
    if require_layers is not None:
        layer_names = list(require_layers)
    else:
        layer_names = sorted(layer_map.keys(), key=_layer_sort_key)

    expected_fas = list(range(1, 194))
    L = len(layer_names)
    T = len(time_vals)
    data = np.empty((T, L, 193), dtype=np.float64)

    for li, layer in enumerate(layer_names):
        if layer not in layer_map:
            _die(f"Layer '{layer}' not found after parsing. (Check column naming.)")
        mapping = layer_map[layer]
        if len(mapping) != 193:
            missing = [fa for fa in expected_fas if fa not in mapping]
            _die(
                f"Layer '{layer}' has {len(mapping)} parsed columns (expected 193). "
                f"Missing examples: {missing[:25]}{' ...' if len(missing) > 25 else ''}"
            )

        cols = [mapping[fa] for fa in expected_fas]
        block = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        data[:, li, :] = block

    return time_vals, data, layer_names



def _align_by_common_times(
    times_list: List[np.ndarray],
    data_list: List[np.ndarray],
    *,
    round_decimals: int = 9,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Align one or more (time, data) series by the intersection of times.

    - Times are rounded to `round_decimals` before intersection (helps with float formatting drift).
    - The aligned arrays are returned in *sorted common-time order* so every dataset uses the same
      row ordering.

    Returns:
      t_aommon: (T_common,)
      aligned_data_list: list of arrays, each shape (T_common, L, 193)
    """
    if len(times_list) != len(data_list) or len(times_list) < 2:
        _die("Alignment requires at least two (time, data) series.")

    keys_list = [np.round(t.astype(float), round_decimals) for t in times_list]

    common = keys_list[0]
    for k in keys_list[1:]:
        common = np.intersect1d(common, k)

    if common.size == 0:
        _die(
            "No overlapping times across the provided CSVs after rounding. "
            f"(round_decimals={round_decimals})"
        )

    # Deterministic ordering for all datasets
    common = np.sort(common)

    aligned = []
    for keys, data in zip(keys_list, data_list):
        # Map rounded time -> first index
        idx_bap: Dict[float, int] = {}
        for i, val in enumerate(keys.tolist()):
            if val not in idx_bap:
                idx_bap[val] = i

        try:
            idxs = np.array([idx_bap[val] for val in common.tolist()], dtype=int)
        except KeyError:
            _die("Alignment failed unexpectedly (time key missing during index mapping).")

        aligned.append(data[idxs, :, :])

    return common, aligned


def compute_error_metrics(
    ref: np.ndarray,
    comp: np.ndarray,
    *,
    eps: float = _PCT_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (max_over_time_pct, time_avg_pct) with shape (L, 193)."""
    if ref.shape != comp.shape:
        _die(f"Metric compute: shape mismatch ref={ref.shape} vs comp={comp.shape}")

    # Max-over-time percent diff per channel
    max_ref = np.nanmax(ref, axis=0)  # (L, 193)
    max_aomp = np.nanmax(comp, axis=0)
    max_pct = (max_aomp - max_ref) / (max_ref + eps) * 100.0

    # Time-avg percent diff per channel (average of instantaneous percent diff)
    inst = (comp - ref) / (ref + eps) * 100.0
    timeavg_pct = np.nanmean(inst, axis=0)

    return max_pct, timeavg_pct


def _sym_limits(vals: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    s = max(abs(vmin), abs(vmax))
    return (-s, s)


def plot_error_base(
    vals_193: np.ndarray,
    *,
    mask: np.ndarray,
    out_png: Path,
    title: str,
    symmetric: bool = True,
) -> None:
    grid = reshape_193_to_15x15(vals_193, mask)
    grid_plot = grid.copy()
    grid_plot[~mask] = np.nan

    if symmetric:
        vmin, vmax = _sym_limits(vals_193)
    else:
        vmin, vmax = float(np.nanmin(vals_193)), float(np.nanmax(vals_193))

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid_plot, interpolation="none", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percent difference (%)")

    # Basic stats annotation
    mn = float(np.nanmean(vals_193))
    mx = float(np.nanmax(vals_193))
    mi = float(np.nanmin(vals_193))
    ma = float(np.nanmax(np.abs(vals_193)))
    txt = f"min={mi:.3g}%  max={mx:.3g}%\nmean={mn:.3g}%  max|.|={ma:.3g}%"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved error plot -> {out_png}")


def plot_error_grid_3x3(
    vals_L193: np.ndarray,
    layer_names: List[str],
    *,
    mask: np.ndarray,
    out_png: Path,
    title: str,
    colorbar_mode: str = "shared",  # "shared" or "per_subplot"
    symmetric: bool = True,
) -> None:
    if vals_L193.shape[0] != len(layer_names) or vals_L193.shape[1] != 193:
        _die(f"Grid plot: expected (L,193) matching layers; got {vals_L193.shape} for {len(layer_names)} layers")

    if colorbar_mode not in ("shared", "per_subplot"):
        _die(f"Grid plot: invalid colorbar_mode='{colorbar_mode}'")

    # Precompute grids
    grids = []
    for i in range(len(layer_names)):
        g = reshape_193_to_15x15(vals_L193[i], mask)
        g[~mask] = np.nan
        grids.append(g)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes_flat = axes.ravel()

    if colorbar_mode == "shared":
        if symmetric:
            vmin, vmax = _sym_limits(vals_L193)
        else:
            vmin, vmax = float(np.nanmin(vals_L193)), float(np.nanmax(vals_L193))

    last_im = None
    for i in range(9):
        ax = axes_flat[i]
        if i < len(layer_names):
            if colorbar_mode == "shared":
                im = ax.imshow(grids[i], interpolation="none", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            else:
                if symmetric:
                    vmin_i, vmax_i = _sym_limits(vals_L193[i])
                else:
                    vmin_i, vmax_i = float(np.nanmin(vals_L193[i])), float(np.nanmax(vals_L193[i]))
                im = ax.imshow(grids[i], interpolation="none", cmap="RdBu_r", vmin=vmin_i, vmax=vmax_i)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label("%", fontsize=8)
            last_im = im
            ax.set_title(layer_names[i])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    if colorbar_mode == "shared" and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_flat.tolist(), shrink=0.8)
        cbar.set_label("Percent difference (%)")

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved error plot -> {out_png}")


def run_error_analysis_2f(
    *,
    comp_csv: Path,
    ref_csv: Path,
    out_dir: Path,
    mask: np.ndarray,
    round_decimals: int = 9,
    symmetric: bool = True,
    case_name: Optional[str] = None,
    comp_label: Optional[str] = None,
    ref_label: Optional[str] = None,
) -> None:
    """Compare comp vs ref and emit 6 plots (Base + 3x3 grids for Max/TimeAvg with shared/per-subplot cbars)."""
    require_layers = ["Base"] + [f"Layer{i}" for i in range(1, 9)]

    comp_df = load_massflow_csv_auto_sep(comp_csv)
    ref_df = load_massflow_csv_auto_sep(ref_csv)

    t_a, x_a, layers_c = standardize_to_layers_193(comp_df, require_layers=require_layers)
    t_r, x_r, layers_r = standardize_to_layers_193(ref_df, require_layers=require_layers)
    t, aligned = _align_by_common_times([t_a, t_r], [x_a, x_r], round_decimals=round_decimals)
    x_a2, x_r2 = aligned

    # Compute errors as (comp - ref) / ref
    max_pct, timeavg_pct = compute_error_metrics(ref=x_r2, comp=x_a2)

    comp_name = comp_label or comp_csv.stem
    ref_name = ref_label or ref_csv.stem
    out = out_dir / f"2f_{comp_name}_vs_{ref_name}"
    out.mkdir(parents=True, exist_ok=True)

    # Base layer index 0
    plot_error_base(
        max_pct[0],
        mask=mask,
        out_png=out / "base_max_pct.png",
        title=_with_case_prefix(f"Base max-over-time % diff: {comp_name} vs {ref_name}", case_name),
        symmetric=symmetric,
    )
    plot_error_base(
        timeavg_pct[0],
        mask=mask,
        out_png=out / "base_timeavg_pct.png",
        title=_with_case_prefix(f"Base time-avg % diff: {comp_name} vs {ref_name}", case_name),
        symmetric=symmetric,
    )

    plot_error_grid_3x3(
        max_pct,
        layer_names=layers_c,
        mask=mask,
        out_png=out / "grid_max_pct_shared.png",
        title=_with_case_prefix(f"Max-over-time % diff (shared cbar): {comp_name} vs {ref_name}", case_name),
        colorbar_mode="shared",
        symmetric=symmetric,
    )
    plot_error_grid_3x3(
        max_pct,
        layer_names=layers_c,
        mask=mask,
        out_png=out / "grid_max_pct_per_subplot.png",
        title=_with_case_prefix(f"Max-over-time % diff (per-subplot cbar): {comp_name} vs {ref_name}", case_name),
        colorbar_mode="per_subplot",
        symmetric=symmetric,
    )
    plot_error_grid_3x3(
        timeavg_pct,
        layer_names=layers_c,
        mask=mask,
        out_png=out / "grid_timeavg_pct_shared.png",
        title=_with_case_prefix(f"Time-avg % diff (shared cbar): {comp_name} vs {ref_name}", case_name),
        colorbar_mode="shared",
        symmetric=symmetric,
    )
    plot_error_grid_3x3(
        timeavg_pct,
        layer_names=layers_c,
        mask=mask,
        out_png=out / "grid_timeavg_pct_per_subplot.png",
        title=_with_case_prefix(f"Time-avg % diff (per-subplot cbar): {comp_name} vs {ref_name}", case_name),
        colorbar_mode="per_subplot",
        symmetric=symmetric,
    )


def run_error_analysis_3f(
    *,
    case_a_csv: Path,
    case_b_csv: Path,
    case_c_csv: Path,
    out_dir: Path,
    mask: np.ndarray,
    round_decimals: int = 9,
    symmetric: bool = True,
    case_name: Optional[str] = None,
    case_a_label: Optional[str] = None,
    case_b_label: Optional[str] = None,
    case_c_label: Optional[str] = None,
) -> None:
    """Compare three case inputs (pairwise) and emit 18 plots, matching the original 3-fidelity script structure."""
    require_layers = ["Base"] + [f"Layer{i}" for i in range(1, 9)]

    df_a = load_massflow_csv_auto_sep(case_a_csv)
    df_b = load_massflow_csv_auto_sep(case_b_csv)
    df_c = load_massflow_csv_auto_sep(case_c_csv)

    t_a, x_a, layers = standardize_to_layers_193(df_a, require_layers=require_layers)
    t_b, x_b, _ = standardize_to_layers_193(df_b, require_layers=require_layers)
    t_c, x_c, _ = standardize_to_layers_193(df_c, require_layers=require_layers)

    # Align all three by common time intersection
    t_all, aligned = _align_by_common_times([t_a, t_b, t_c], [x_a, x_b, x_c], round_decimals=round_decimals)
    x_a3, x_b3, x_c3 = aligned

    name_a = case_a_label or case_a_csv.stem
    name_b = case_b_label or case_b_csv.stem
    name_c = case_c_label or case_c_csv.stem

    out = out_dir / f"3f_{name_c}_{name_b}_{name_a}"
    out.mkdir(parents=True, exist_ok=True)

    pairs = [
        (name_c, x_a3, name_a, x_a3),
        (name_b, x_b3, name_a, x_a3),
        (name_c, x_a3, name_b, x_b3),
    ]

    for comp_name, comp_x, ref_name, ref_x in pairs:
        max_pct, timeavg_pct = compute_error_metrics(ref=ref_x, comp=comp_x)

        for mode_name, vals in [("max", max_pct), ("timeavg", timeavg_pct)]:
            # Base
            plot_error_base(
                vals[0],
                mask=mask,
                out_png=out / f"base_{mode_name}_{comp_name}_vs_{ref_name}.png",
                title=_with_case_prefix(f"Base {mode_name} % diff: {comp_name} vs {ref_name}", case_name),
                symmetric=symmetric,
            )
            # Grids
            plot_error_grid_3x3(
                vals,
                layer_names=layers,
                mask=mask,
                out_png=out / f"grid_{mode_name}_{comp_name}_vs_{ref_name}_shared.png",
                title=_with_case_prefix(f"{mode_name} % diff (shared cbar): {comp_name} vs {ref_name}", case_name),
                colorbar_mode="shared",
                symmetric=symmetric,
            )
            plot_error_grid_3x3(
                vals,
                layer_names=layers,
                mask=mask,
                out_png=out / f"grid_{mode_name}_{comp_name}_vs_{ref_name}_per_subplot.png",
                title=_with_case_prefix(f"{mode_name} % diff (per-subplot cbar): {comp_name} vs {ref_name}", case_name),
                colorbar_mode="per_subplot",
                symmetric=symmetric,
            )
# =============================================================================
# Output directory management
# =============================================================================


def infer_case_label_from_path(path: Path) -> Optional[str]:
    """Infer case label from a path segment named like FApitch_overX."""
    for part in path.parts:
        if _CASE_DIR_RE.fullmatch(part):
            return part
    return None


def validate_batches_dir_and_case(batches_dir: Path) -> Tuple[Path, str]:
    """Validate raw_csv input layout and return (case_dir, case_name).

    Required format: <...>/FApitch_overX/raw_csv
    """
    if not batches_dir.exists():
        _die(f"Batches directory not found: {batches_dir}")
    if not batches_dir.is_dir():
        _die(f"Batches path must be a directory: {batches_dir}")
    if batches_dir.name != "raw_csv":
        _die(
            "--batches-dir must point to a 'raw_csv' directory. "
            f"Got: {batches_dir}"
        )

    case_dir = batches_dir.parent
    case_name = case_dir.name
    if not _CASE_DIR_RE.fullmatch(case_name):
        _die(
            "Parent directory of raw_csv must match 'FApitch_overX' (X integer). "
            f"Got: {case_name}"
        )

    return case_dir, case_name


def resolve_reordered_csv_from_case_dir(case_dir: Path) -> Path:
    """Resolve reordered CSV from a validated FApitch_overX case directory."""
    if not case_dir.exists() or not case_dir.is_dir():
        _die(f"Case directory not found: {case_dir}")
    if not _CASE_DIR_RE.fullmatch(case_dir.name):
        _die(
            "Case directory must match 'FApitch_overX' (X integer). "
            f"Got: {case_dir.name}"
        )

    inter_dir = case_dir / "intermediate"
    if not inter_dir.exists() or not inter_dir.is_dir():
        _die(f"Missing intermediate directory in case path: {inter_dir}")

    preferred = inter_dir / f"{case_dir.name}_reordered.csv"
    if preferred.exists():
        return preferred

    legacy = inter_dir / "reordered.csv"
    if legacy.exists():
        return legacy

    candidates = sorted(inter_dir.glob("*_reordered.csv"), key=lambda x: x.name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        _die(
            f"Multiple *_reordered.csv files found in {inter_dir}; cannot choose automatically. "
            "Keep only one candidate or use the canonical case-named file."
        )

    _die(f"No reordered CSV found in {inter_dir}. Expected {preferred.name} or reordered.csv")


def preferred_csv_name(kind: str, case_label: Optional[str]) -> str:
    """Build output CSV filenames.

    Examples:
      preferred_csv_name("reordered", "FApitch_over12") ->
          "FApitch_over12_reordered.csv"
      preferred_csv_name("reordered", None) -> "reordered.csv"
    """
    base = f"{kind}.csv"
    return f"{case_label}_{base}" if case_label else base


def is_reordered_csv_name(name: str) -> bool:
    """Accept both legacy and case-aware reordered CSV names."""
    low = name.lower()
    return low == "reordered.csv" or low.endswith("_reordered.csv")


def infer_processing_out_dir_from_batches(args: argparse.Namespace) -> Optional[Path]:
    """Infer output run directory from --batches-dir (strict raw_csv layout)."""
    if not args.batches_dir:
        return None
    case_dir, _ = validate_batches_dir_and_case(Path(args.batches_dir))
    return case_dir


def _with_case_prefix(title: str, case_name: Optional[str]) -> str:
    return f"[{case_name}] {title}" if case_name else title


def make_output_dir(run_label: str, overwrite: bool) -> Path:
    date_str = _dt.datetime.now().strftime("%m%d%Y")
    out_dir = Path("outputs") / f"{run_label}_{date_str}"

    if out_dir.exists():
        if not overwrite:
            _die(
                f"Output directory already exists: {out_dir}\n"
                "Refusing to overwrite. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# CLI + main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mf_pipeline.py",
        description="Combine batch CSVs, reorder by layer, convert to HDF5 planes, and generate mapping grid plots.",
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--batches",
        nargs="+",
        help="Explicit ordered list of batch CSV files to combine.",
    )
    src.add_argument(
        "--batches-dir",
        type=str,
        help="Required for processing: path to the case raw CSV folder (<...>/FApitch_overX/raw_csv).",
    )

    p.add_argument(
        "--pattern",
        default="batch*.csv",
        help="Glob pattern used to discover batch files when using --batches-dir (default: batch*.csv).",
    )

    p.add_argument(
        "--run-label",
        type=str,
        default="analysis",
        help="Label used for fallback output folder naming when a case folder cannot be inferred (default: analysis).",
    )

    p.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Minimum time (inclusive) for selecting rows from the batch CSVs.",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Maximum time (inclusive) for selecting rows from the batch CSVs.",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the auto-named output directory if it already exists.",
    )

    p.add_argument(
        "--video",
        action="store_true",
        help="After writing reordered.csv, generate BOTH core heatmap videos (shared colorbar and per-subplot colorbars; requires OpenCV).",
    )
    p.add_argument(
        "--video-fps",
        type=int,
        default=60,
        help="Frames per second for the output video (default: 60).",
    )
    p.add_argument(
        "--video-vmin",
        type=float,
        default=None,
        help="Optional fixed vmin for video color scale. If omitted, uses data min.",
    )
    p.add_argument(
        "--video-vmax",
        type=float,
        default=None,
        help="Optional fixed vmax for video color scale. If omitted, uses data max.",
    )
    p.add_argument(
        "--video-max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of rendered frames (applies after --video-every).",
    )
    p.add_argument(
        "--video-keep-frames",
        action="store_true",
        help="Save each rendered frame as a PNG alongside the MP4 (default: False).",
    )
    p.add_argument(
        "--video-every",
        type=int,
        default=20,
        help="Render only every Nth timestep to reduce video length (default: 20).",
    )
    p.add_argument(
        "--custom-cmap",
        action="store_true",
        help="Use the custom 32-color colormap (legacy custom script behavior).",
    )


    # ---------------------------
    # Analysis-only / existing artifacts
    # ---------------------------
    p.add_argument(
        "--use-reordered",
        type=str,
        default=None,
        help="Skip batch processing and use an existing reordered.csv for video/error analysis.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory override (useful for analysis-only runs). If omitted, defaults to the normal outputs/<run_label>_<date> folder, or the parent outputs folder inferred from --use-reordered when possible.",
    )

    # ---------------------------
    # Step 4 (optional): errors
    # ---------------------------
    p.add_argument(
        "--core-mask-npy",
        type=str,
        default=None,
        help="Optional path to a 15x15 boolean core mask .npy file with 193 True cells. If omitted, uses the built-in Catawba-style mask.",
    )

    p.add_argument(
        "--error-2f",
        action="store_true",
        help="Run 2-fidelity error analysis between two case directories specified by --error-2f-comp-dir and --error-2f-ref-dir.",
    )
    p.add_argument(
        "--error-2f-comp-dir",
        type=str,
        default=None,
        help="Comparison case directory for --error-2f (must be named FApitch_overX and contain intermediate/*_reordered.csv).",
    )
    p.add_argument(
        "--error-2f-ref-dir",
        type=str,
        default=None,
        help="Reference case directory for --error-2f (must be named FApitch_overX and contain intermediate/*_reordered.csv).",
    )

    p.add_argument(
        "--error-3f",
        action="store_true",
        help="Run 3-fidelity error analysis between three case directories: --error-3f-case-a-dir/--error-3f-case-b-dir/--error-3f-case-c-dir.",
    )
    p.add_argument("--error-3f-case-a-dir", type=str, default=None, help="First case directory for --error-3f (FApitch_overX with intermediate/*_reordered.csv).")
    p.add_argument("--error-3f-case-b-dir", type=str, default=None, help="Second case directory for --error-3f (FApitch_overX with intermediate/*_reordered.csv).")
    p.add_argument("--error-3f-case-c-dir", type=str, default=None, help="Third case directory for --error-3f (FApitch_overX with intermediate/*_reordered.csv).")

    p.add_argument(
        "--error-time-round",
        type=int,
        default=9,
        help="Decimal places to round time values to before taking intersections (default: 9).",
    )
    p.add_argument(
        "--error-asymmetric",
        action="store_true",
        help="Use raw min/max color limits for error plots (default uses symmetric +/- limits).",
    )

    return p



def main() -> None:
    args = build_parser().parse_args()

    # ---------------------------------------
    # Decide whether we need to run processing
    # ---------------------------------------
    have_batches = bool(args.batches_dir) or bool(getattr(args, "batches", None))
    have_use_reordered = args.use_reordered is not None
    have_3f_inputs = bool(args.error_3f) and bool(args.error_3f_case_a_dir) and bool(args.error_3f_case_b_dir) and bool(args.error_3f_case_c_dir)
    have_2f_inputs = bool(args.error_2f) and bool(args.error_2f_comp_dir) and bool(args.error_2f_ref_dir)

    need_processing = (not have_use_reordered) and (have_batches)

    if need_processing and not args.batches_dir:
        _die("Processing mode requires --batches-dir pointing to a raw_csv directory.")
    if need_processing and getattr(args, "batches", None):
        _die("--batches is no longer supported for processing. Use --batches-dir <.../FApitch_overX/raw_csv>.")
    if need_processing and args.out_dir:
        _die("In processing mode, outputs are fixed to the parent case directory of --batches-dir/raw_csv.")

    if not need_processing and not have_use_reordered and not have_3f_inputs and not have_2f_inputs:
        _die(
            "Nothing to do: provide --batches-dir to process new CSVs, "
            "or --use-reordered to analyze an existing reordered.csv, "
            "or --error-2f with --error-2f-comp-dir/--error-2f-ref-dir, "
            "or --error-3f with --error-3f-case-a-dir/--error-3f-case-b-dir/--error-3f-case-c-dir."
        )

    # ----------------------
    # Determine output folder
    # ----------------------
    out_dir: Path
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    elif have_use_reordered:
        comp_path = Path(args.use_reordered)
        # If the path looks like .../<run>/intermediate/*_reordered.csv, infer <run> as out_dir.
        if is_reordered_csv_name(comp_path.name) and comp_path.parent.name.lower() == "intermediate":
            out_dir = comp_path.parent.parent
        else:
            # Fallback: create a dated analysis folder under ./outputs
            stamp = datetime.datetime.now().strftime("%m%d%Y")
            out_dir = Path("outputs") / f"analysis_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

    elif have_3f_inputs:
        case_c_path = Path(args.error_3f_case_c_dir)
        if _CASE_DIR_RE.fullmatch(case_c_path.name):
            out_dir = case_c_path
        # If the path looks like .../<run>/intermediate/*_reordered.csv, infer <run> as out_dir.
        elif is_reordered_csv_name(case_c_path.name) and case_c_path.parent.name.lower() == "intermediate":
            out_dir = case_c_path.parent.parent
        else:
            stamp = datetime.datetime.now().strftime("%m%d%Y")
            out_dir = Path("outputs") / f"analysis_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        inferred = infer_processing_out_dir_from_batches(args) if need_processing else None
        if inferred is not None:
            out_dir = inferred
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = make_output_dir(args.run_label, overwrite=args.overwrite)

    case_name: Optional[str] = None
    if need_processing:
        _, case_name = validate_batches_dir_and_case(Path(args.batches_dir))
    elif have_use_reordered:
        case_name = infer_case_label_from_path(Path(args.use_reordered))
    elif have_2f_inputs:
        case_name = None
    elif have_3f_inputs:
        case_name = None

    # ----------------------
    # Processing path
    # ----------------------
    if need_processing:
        batch_paths = discover_batch_files(Path(args.batches_dir), args.pattern)

        inter_dir = out_dir / "intermediate"
        hdf5_dir = out_dir / "hdf5"
        plots_dir = out_dir / "plots"

        combined_csv = combine_batches(
            batch_paths=batch_paths,
            out_csv=inter_dir / "combined.csv",
            t_min=args.t_min,
            t_max=args.t_max,
        )

        reordered_csv = reorder_csv_group_by_layer(
            in_csv=combined_csv,
            out_csv=inter_dir / "reordered.csv",
        )

        csv_to_hdf5_planes_and_plots(
            in_csv=reordered_csv,
            hdf5_dir=hdf5_dir,
            plots_dir=plots_dir,
            case_name=case_name,
            use_custom_cmap=args.custom_cmap,
        )
    else:
        reordered_csv = Path(args.use_reordered) if have_use_reordered else None

    # ----------------------
    # Optional video
    # ----------------------
    if args.video:
        if reordered_csv is None:
            _die("--video requires either processing inputs (--batches/--batches-dir) or --use-reordered.")

        for mode in ("shared", "per_subplot"):
            generate_video_from_reordered_csv(
                reordered_csv=reordered_csv,
                video_dir=out_dir / "video",
                case_name=case_name,
                fps=args.video_fps,
                vmin=args.video_vmin,
                vmax=args.video_vmax,
                max_frames=args.video_max_frames,
                keep_frames=args.video_keep_frames,
                every_n=max(1, int(args.video_every)),
                colorbar_mode=mode,
                use_custom_cmap=args.custom_cmap,
            )

    # ----------------------
    # Optional error analyses
    # ----------------------
    if args.error_2f:
        if not (args.error_2f_comp_dir and args.error_2f_ref_dir):
            _die("--error-2f requires --error-2f-comp-dir and --error-2f-ref-dir (case dirs named FApitch_overX).")

        comp_case_dir = Path(args.error_2f_comp_dir)
        ref_case_dir = Path(args.error_2f_ref_dir)

        if not _CASE_DIR_RE.fullmatch(comp_case_dir.name):
            _die(f"--error-2f-comp-dir must be a case directory named FApitch_overX. Got: {comp_case_dir.name}")
        if not _CASE_DIR_RE.fullmatch(ref_case_dir.name):
            _die(f"--error-2f-ref-dir must be a case directory named FApitch_overX. Got: {ref_case_dir.name}")

        comp_csv_2f = resolve_reordered_csv_from_case_dir(comp_case_dir)
        ref_csv_2f = resolve_reordered_csv_from_case_dir(ref_case_dir)

        err_mask = np.load(args.core_mask_npy).astype(bool) if args.core_mask_npy else core_mask_15x15()
        run_error_analysis_2f(
            comp_csv=comp_csv_2f,
            ref_csv=ref_csv_2f,
            out_dir=out_dir / "error_plots",
            mask=err_mask,
            round_decimals=int(args.error_time_round),
            symmetric=(not args.error_asymmetric),
            case_name=None,
            comp_label=comp_case_dir.name,
            ref_label=ref_case_dir.name,
        )

    if args.error_3f:
        if not (args.error_3f_case_a_dir and args.error_3f_case_b_dir and args.error_3f_case_c_dir):
            _die("--error-3f requires --error-3f-case-a-dir, --error-3f-case-b-dir, and --error-3f-case-c-dir (case dirs named FApitch_overX).")

        case_a_dir = Path(args.error_3f_case_a_dir)
        case_b_dir = Path(args.error_3f_case_b_dir)
        case_c_dir = Path(args.error_3f_case_c_dir)

        for flag, case_dir in [
            ("--error-3f-case-a-dir", case_a_dir),
            ("--error-3f-case-b-dir", case_b_dir),
            ("--error-3f-case-c-dir", case_c_dir),
        ]:
            if not _CASE_DIR_RE.fullmatch(case_dir.name):
                _die(f"{flag} must be a case directory named FApitch_overX. Got: {case_dir.name}")

        case_a_csv_3f = resolve_reordered_csv_from_case_dir(case_a_dir)
        case_b_csv_3f = resolve_reordered_csv_from_case_dir(case_b_dir)
        case_c_csv_3f = resolve_reordered_csv_from_case_dir(case_c_dir)

        err_mask = np.load(args.core_mask_npy).astype(bool) if args.core_mask_npy else core_mask_15x15()
        run_error_analysis_3f(
            case_a_csv=case_a_csv_3f,
            case_b_csv=case_b_csv_3f,
            case_c_csv=case_c_csv_3f,
            out_dir=out_dir / "error_plots",
            mask=err_mask,
            round_decimals=int(args.error_time_round),
            symmetric=(not args.error_asymmetric),
            case_name=None,
            case_a_label=case_a_dir.name,
            case_b_label=case_b_dir.name,
            case_c_label=case_c_dir.name,
        )


if __name__ == "__main__":
    main()
