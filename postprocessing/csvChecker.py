#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import h5py

# ─── CONFIG ────────────────────────────────────────────────────────────────
MASTER_XYZ_PATH = 'masterXYZ.csv'    # path to your master XYZ file
CSV_DIR         = '.'                # directory containing the CSVs to process
OUTPUT_DIR      = 'h5_output'        # where to write the HDF5 files

# ─── MAIN SCRIPT ───────────────────────────────────────────────────────────
def main():
    # 1) Load master XYZ coordinates
    master_df = pd.read_csv(MASTER_XYZ_PATH)
    master_coords = master_df[['X (in)', 'Y (in)', 'Z (in)']].to_numpy()

    # 2) Find all CSVs except the master
    all_csvs = glob.glob(os.path.join(CSV_DIR, '*.csv'))
    to_process = [p for p in all_csvs if os.path.abspath(p) != os.path.abspath(MASTER_XYZ_PATH)]
    if not to_process:
        print("No CSVs found to process.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3) Process each CSV
    for csv_path in to_process:
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path)

        # 3a) Verify coordinate columns match master
        coords = df[['X (in)', 'Y (in)', 'Z (in)']].to_numpy()
        if coords.shape != master_coords.shape or not np.array_equal(coords, master_coords):
            raise ValueError(f"Mesh ordering mismatch in {csv_path} — aborting.")

        # 3b) Drop the coordinate columns and overwrite CSV
        df_props = df.drop(['X (in)', 'Y (in)', 'Z (in)'], axis=1)
        df_props.to_csv(csv_path, index=False)
        print("Coordinates dropped, CSV overwritten.")

        # 3c) Convert to HDF5
        data_array = df_props.to_numpy()
        base_name  = os.path.splitext(os.path.basename(csv_path))[0]
        h5_path    = os.path.join(OUTPUT_DIR, f"{base_name}.h5")
        with h5py.File(h5_path, 'w') as h5:
            h5.create_dataset('properties', data=data_array)
        print(f"Saved properties to {h5_path} (shape={data_array.shape})")

    # 4) Cleanup: delete all CSVs except the master
    for csv_path in to_process:
        os.remove(csv_path)
        print(f"Deleted intermediate CSV: {csv_path}")

    print("\nAll done. Remaining CSV only:", MASTER_XYZ_PATH)
    print("HDF5 files in directory:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
