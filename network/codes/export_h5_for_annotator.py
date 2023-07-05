# Modules
import os, sys

import numpy as np

from tqdm import tqdm
from h5py import File as h5File
import onnxruntime


# Parameters
fp_onnx_model = "PATH_TO_ONNX_MODEL_FILE"
fp_data = "PATH_TO_BEHAVIOR_CAMERA_H5_FILE"
fp_folder_path_output = "PATH_TO_DESIRED_FOLDER_FOR_EXPORT"



# Main
if __name__ == '__main__':
    # Use arguments if available
    if len(sys.argv) > 1:
        fp_onnx_model = sys.argv[1]
    if len(sys.argv) > 2:
        fp_data = sys.argv[2]
    if len(sys.argv) > 3:
        fp_folder_path_output = sys.argv[3]
    
    # Creat results folder
    if not os.path.exists(fp_folder_path_output):
        os.mkdir(fp_folder_path_output)
    fp_annotations = os.path.join( fp_folder_path_output, "annotations.h5" )
    fp_worldlines = os.path.join( fp_folder_path_output, "worldlines.h5" )

    # Load Runtime
    ort_session = onnxruntime.InferenceSession(fp_onnx_model)
    
    # Connect to Data
    file_data = h5File(fp_data)
    T, nx, ny = file_data['data'].shape

    # Worldlines
    worldline_color = b"#ffffff"
    worldline_id = 0
    worldline_name = b"PHARYNX"
    with h5File(fp_worldlines, mode='w') as file_worldlines:
        file_worldlines['id'] = np.array([ worldline_id ])
        file_worldlines['name'] = np.array([ worldline_name ])
        file_worldlines['color'] = np.array([ worldline_color ])

    # Annotations
    xs_model, ys_model = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    ## Get Predictions from Model
    CHUNK_SIZE = 128
    N_CHUNKS = (T//CHUNK_SIZE) + 1
    for i_chunk in tqdm(range(N_CHUNKS), desc=f"Converting data (batch_size={CHUNK_SIZE}) progress ... "):
        # Load Images & Crop
        il = i_chunk*CHUNK_SIZE
        ir = il + CHUNK_SIZE
        images = file_data['data'][ il:ir, 56:-56, 56:-56 ]
        batch = {
            'input': np.repeat(
                images[:, None, :, :], 3, 1
            ).astype(np.float32)
        }
        # Outputs
        ort_out = ort_session.run(None, batch)
        x_model, y_model = ort_out[0].T
        # Store
        xs_model[il:ir] = (x_model+56)/nx
        ys_model[il:ir] = (y_model+56)/ny
    ## Write file
    with h5File(fp_annotations, mode='w') as file_annotations:
        file_annotations['id'] = 1 + np.arange(T)
        file_annotations['parent_id'] = 1 + np.arange(T)
        file_annotations['provenance'] = np.array([b'NNET']*T)
        file_annotations['t_idx'] = np.arange(T)
        file_annotations['x'] = xs_model
        file_annotations['y'] = ys_model
        file_annotations['z'] = np.zeros(T, dtype=np.float32)
        file_annotations['worldline_id'] = np.array([worldline_id]*T)