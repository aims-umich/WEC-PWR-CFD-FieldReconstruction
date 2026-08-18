import h5py
import numpy as np


def list_hdf5_datasets(h5obj, prefix: str = "") -> list[str]:
    paths = []
    if isinstance(h5obj, h5py.Dataset):
        paths.append(prefix.rstrip("/"))
    elif isinstance(h5obj, h5py.Group):
        for key, value in h5obj.items():
            paths.extend(list_hdf5_datasets(value, f"{prefix}/{key}"))
    return paths



def load_first_matching_txhxw(path, expected_h, expected_w, dtype=np.float32):
    with h5py.File(path, "r") as h5:
        dataset_paths = list_hdf5_datasets(h5, "")
        for dataset_path in dataset_paths:
            dataset = h5[dataset_path]
            if dataset.ndim == 3 and dataset.shape[1] == expected_h and dataset.shape[2] == expected_w:
                return dataset[...].astype(dtype), dataset_path
        fallback_path = dataset_paths[0]
        return h5[fallback_path][...].astype(dtype), fallback_path



def load_multiple_levels(paths, expected_h, expected_w, dtype=np.float32):
    arrays = []
    dataset_paths = []
    for path in paths:
        array, dataset_path = load_first_matching_txhxw(path, expected_h, expected_w, dtype=dtype)
        arrays.append(array)
        dataset_paths.append(dataset_path)
    return arrays, dataset_paths
