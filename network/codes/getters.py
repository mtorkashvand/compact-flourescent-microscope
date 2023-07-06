import h5py
import numpy as np
from pathlib import Path

"""
To be able to open the data with annotator app, put a copy of this
script inside the folder containng the data. rename the data to 'data.h5'
"""

def get_slice(dataset: Path, t: int) -> np.ndarray:
    h5_filename = dataset / "data.h5"
    f = h5py.File(h5_filename, 'r')
    return np.repeat(f["data"][t][None, None, :, :], 3, 0)


def get_metadata(dataset: Path) -> dict:
    h5_filename = dataset / "data.h5"
    f = h5py.File(h5_filename, 'r')
    metadata = {
        "shape_t" : len(f["data"]),
        "shape_c" : 3,
        "shape_z" : 1,
        "shape_y" : 512,
        "shape_x" : 512,
        "dtype": "np.uint8"
    }
    return metadata
