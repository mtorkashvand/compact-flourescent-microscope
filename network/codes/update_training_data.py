import os
import json
from math import floor

import h5py
import numpy as np
from tqdm import tqdm

def update_training_data(src, dest=None, condition='plate'):
    """
    Updates training data located in 'dest' with annotations found in 'dest'.

    Assumptions:
    1. The file containing the data in the src directory is named 'data.h5'
    2. 'data.h5' has two groups: 'data' and 'times'
    3. the frame format saved in the group 'data' is: [Y, X]
    4. there is an 'annotations.h5' file in the src directory.
    5. 'annotations.h5' has goups named 't_idx', 'x' and 'y' and 'worldline_id
    6. 'x' and 'y' in the 'annotations.h5' file are normalized coords [0, 1]
    7. 'worldline_id' == 1 corresponds to background
    Note:
        These assumptions are to make it compatible with the output of the annotator app.

    Parameters:
        src:
            - the directory of the input 'data.h5' file, it is assumed that the 
              'annotations.h5' file is also in this directory.
        dest:
            - the directory in which the existing 'training_data.h5' can be found.
              if the mentioned file does not exist, it will be created.
        condition:
            - specifies the first dataset index. first index is 0 for 'plate' and
              1000 for 'glass'

    The output file, 'training_data.h5' has two groups: 'data' and 'annotations'
    """
    if dest is None:
        dest = r'V:\Mahdi\OpenAutoScope2.0\data\training_data'
    
    if os.path.exists(os.path.join(dest, 'dataset_paths.json')):
        with open(os.path.join(dest, 'dataset_paths.json'), 'r') as f:
            appended_datasets = json.load(f)
    else:
        appended_datasets = {}

    appended_datasets_paths = list(appended_datasets.values())
    if src in appended_datasets_paths:
        return

    try:
        src_file = h5py.File(os.path.join(src, 'data.h5'), 'r')
    except:
        print("Error: No data.h5 file found")
        return
    
    try:
        annot_file = h5py.File(os.path.join(src, 'annotations.h5'), 'r')
    except:
        print("Error: No annotations.h5 file found")
        src_file.close()
        return
    
    try:
        sample_slice = src_file["data"][0]
        shape_y, shape_x = sample_slice.shape
    except:
        print("Error: src file is empty.")
        src_file.close()
        annot_file.close()
        return
    

    if os.path.exists(os.path.join(dest, 'training_data.h5')):
        dest_file = h5py.File(os.path.join(dest, 'training_data.h5'), 'r+')
        n_datapoints = len(dest_file['data'])
        data = dest_file["data"]
        annotations = dest_file["annotations"]
        dataset_index = dest_file["index"]
        indices = dataset_index[:]

        if condition == "plate":
            plate_indices = indices[indices<1000]
            if len(plate_indices) == 0:
                index = 0
            else:
                index = np.max(plate_indices) + 1
        else:
            glass_indices = indices[indices>=1000]
            if len(glass_indices) == 0:
                index = 1000
            else:
                index = np.max(glass_indices) + 1

    else:
        dest_file = h5py.File(os.path.join(dest, 'training_data.h5'), 'w-')
        group = dest_file["/"]
        data = group.create_dataset(
            'data',
            (0, shape_y, shape_x),
            chunks=(1, shape_y, shape_x),
            dtype=np.uint8,
            compression="lzf",
            compression_opts=None,
            maxshape=(None, shape_y, shape_x)
        )
        annotations = group.create_dataset(
            'annotations',
            (0, 2),
            chunks=(1, 2),
            dtype=np.int16,
            compression="lzf",
            compression_opts=None,
            maxshape=(None, 2)
        )
        dataset_index = group.create_dataset(
            'index',
            (0, 1),
            chunks=(1, 1),
            dtype=np.uint16,
            compression="lzf",
            compression_opts=None,
            maxshape=(None, 1)
        )
        index = 0 if condition == 'plate' else 1000
        n_datapoints = 0
        

    ts, indices, counts = np.unique(annot_file["t_idx"][:], return_index=True, return_counts=True)

    if len(ts) != 0:
        for t, idx, count in tqdm(zip(ts, indices, counts), desc="Appending new annotated frames "):
            if count == 1:
                y = max(floor(annot_file["y"][idx] * shape_y - 1e-6), 0)
                x = max(floor(annot_file["x"][idx] * shape_x - 1e-6), 0)
                n_datapoints += 1
                data.resize((n_datapoints, shape_y, shape_x))
                annotations.resize((n_datapoints, 2))
                dataset_index.resize((n_datapoints, 1))
                data[n_datapoints-1, :] = src_file["data"][t][:]
                dataset_index[n_datapoints-1, 0] = index
                if int(annot_file["worldline_id"][idx]) == 1:
                    annotations[n_datapoints-1, :] = [-1, -1]
                else:
                    annotations[n_datapoints-1, :] = [y, x]
    else:
        print("Error: Annotations file is empty.")

    appended_datasets[int(index)] = src
    json_appended_datasets = json.dumps(appended_datasets, indent=4)
    with open(os.path.join(dest, 'dataset_paths.json'), 'w') as f:
        f.write(json_appended_datasets)


    src_file.close()
    annot_file.close()
    dest_file.close()
    return