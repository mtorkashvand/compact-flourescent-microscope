# Modules
import numpy as np

from tqdm import tqdm
import datetime
from h5py import File as h5File

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv

# Parameters

# Methods
## Apply OpenCV AffineRotation to a single point
def rotate_xy(M, x, y):
    return np.matmul(
        M,
        np.array([x, y, 1])
    )

# DataLoaders
class AnnotatedDataLoader(Dataset):
    # Constructor
    def __init__(
            self,
            fp_annotated_data,
            factor_augmentations = 1,
            windowsize_min_included = 120,
            crop_size = 400,
            gamma_max_deviation = 0.2,
            always_same_rows = True,
            verbose = False
        ):
        super().__init__()
        self.fp_annotated_data = fp_annotated_data
        self._data_file = h5File(self.fp_annotated_data)
        self.images = self._data_file['data']
        self.nrecords, self.ny, self.nx = self.images.shape
        self.coordinates = self._data_file['annotations']
        self.factor_augmentations = factor_augmentations
        self.n = self.nrecords * self.factor_augmentations
        self.windowsize_min_included = windowsize_min_included
        self.windowsize_min_included2 = self.windowsize_min_included//2
        self.crop_size = crop_size
        self.verbose = verbose

        self.gamma_max_deviation = gamma_max_deviation
        self.always_same_rows = always_same_rows
        return
    # Length
    def __len__(self):
        return self.n
    # Get Item
    def __getitem__(self, i):
        # DEBUG
        if self.always_same_rows:
            np.random.seed(i)
        # Parameters
        idx = i // self.factor_augmentations
        # Choose randomly from 4 possible rotations: 0, 90, 180, 270 degrees
        theta = np.random.choice([ 0, 90, 180, 270 ])
        nx, ny = self.nx, self.ny
        img_processed = self.images[idx]
        y_idx, x_idx = self.coordinates[idx]
        label = 1
        if y_idx >= ny or x_idx >= nx:
            label = 0
            y_idx, x_idx = ny//2, nx//2
        # Gamma
        gamma = 1.0 + (np.random.rand()-0.5)*2*self.gamma_max_deviation
        img_processed = ( (img_processed/255)**gamma * 255 ).astype(np.uint8)
        
        M_rotation = cv.getRotationMatrix2D((nx//2, ny//2), theta, 1.0)
        img_processed = cv.warpAffine(
            img_processed,
            M_rotation,
            (self.nx, self.ny),
            borderValue=255
        )
        coords_new = rotate_xy(M_rotation, x_idx, y_idx).astype(np.int32)
        
        # If Annotation is close to the edge, discard the "Minimum Window Criteria"
        # windowsize_min_included2 = WINDOWSIZE_MIN_INCLUDED2
        # if np.any(coords_new-windowsize_min_included2 < 0) or np.any(coords_new+windowsize_min_included2 >= np.array([nx, ny])):
        #     windowsize_min_included2 = 0
        #     if DEBUG:
        #         print(f"Annotation close to edge encountered! Image IDX: {idx} Label: {label}, Coords: {coords_new}")
        
        # TODO add passing conditions and handle missing annotation inside the frame
        for _ in range(3):
            # Annotation Outside after Rotation
            if np.any(coords_new >= np.array([nx, ny])) or np.any(coords_new < np.zeros(2)):
                if self.verbose:
                    print("Rotation@annotation outside image: {}-{} , shape:({},{})".format(
                        *coords_new,
                        nx, ny
                    ))
                continue
            if self.verbose:
                print(f"Image IDX: {idx} Label: {label}, COORD NEW: {coords_new}")
            x_idx_new, y_idx_new = coords_new
            # If Conditions
            ## X
            if x_idx_new <= self.windowsize_min_included2:
                x_idx_min, x_idx_max = 0, 0
            elif x_idx_new >= (nx - self.windowsize_min_included2):
                x_idx_min, x_idx_max = nx-self.crop_size, nx-self.crop_size
            elif x_idx_new <= nx//2:
                x_idx_min, x_idx_max = 0, min(x_idx_new - self.windowsize_min_included2, nx-self.crop_size)
            elif x_idx_new > nx//2:
                x_idx_min, x_idx_max = max(0, x_idx_new+self.windowsize_min_included2-self.crop_size), nx-self.crop_size
            else:
                raise NotImplementedError()
            ## Y
            if y_idx_new <= self.windowsize_min_included2:
                y_idx_min, y_idx_max = 0, 0
            elif y_idx_new >= (ny - self.windowsize_min_included2):
                y_idx_min, y_idx_max = ny-self.crop_size, ny-self.crop_size
            elif y_idx_new <= ny//2:
                y_idx_min, y_idx_max = 0, min(y_idx_new - self.windowsize_min_included2, ny-self.crop_size)
            elif y_idx_new > ny//2:
                y_idx_min, y_idx_max = max(0, y_idx_new+self.windowsize_min_included2-self.crop_size), ny-self.crop_size
            else:
                raise NotImplementedError()
            if self.verbose:
                print("IDX X MIN-MAX: {}-{} , Y MIN-MAX: {}-{}".format(
                    x_idx_min, x_idx_max,
                    y_idx_min, y_idx_max
                ))
            # Empty Cropping Area
            if x_idx_max < x_idx_min or y_idx_max < y_idx_min:
                if self.verbose:
                    print("Empty crop area: {}-{} , {}-{}".format(
                        x_idx_min, x_idx_max,
                        y_idx_min, y_idx_max
                    ))
                continue
            if self.verbose:
                print(f"{x_idx_min}-{x_idx_max} , {y_idx_min}-{y_idx_max}")
            crop_topleft = np.random.randint(
                (y_idx_min, x_idx_min),
                (y_idx_max+1,x_idx_max+1)
            )  # so `x` can be used for index `i` in slicing and `y` for column indexing
            if self.verbose:
                print(f"{crop_topleft}")
            # Apply
            imin, jmin = crop_topleft
            imax, jmax = crop_topleft+self.crop_size
            img_processed = img_processed[imin:imax, jmin:jmax]
            coords_new -= crop_topleft[::-1]
            # Return
            if self.verbose:
                print(f"Image IDX: {idx} Label: {label}, COORD NEW Cropped: {coords_new}")
            return img_processed, coords_new, label
        print("DEBUG: Attempts to Augment failed!")
        print(f"Image IDX: {idx} Label: {label}, Coords: {coords_new}")
        raise NotImplementedError()

def collate_fn_3d_input(data):
    images, coords, _ = zip(*data)
    coords = np.array(coords)
    images = np.repeat(
        np.array(images)[:,None,:,:], 3, axis=1
    )
    images_channeled = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    return images_channeled, coords
def collate_fn_heatmap(data):
    images, coords, labels = zip(*data)
    _img = images[0]
    images = np.array(images)[:,None,:,:]
    images = torch.tensor( images, dtype=torch.float32 )
    heatmaps = []
    for (i,j), label in zip(coords,labels):
        if label != 0:
            img_annotated = np.zeros_like( _img, dtype=np.float32 )
            img_annotated[j,i] = 100.0
            img_annotated = cv.GaussianBlur(img_annotated,(11,11), 0)
            img_annotated = cv.resize(img_annotated, (IMG_OUTPUT_DIM, IMG_OUTPUT_DIM))
            img_annotated /= img_annotated.max()
        else:
            img_annotated = np.zeros( (IMG_OUTPUT_DIM, IMG_OUTPUT_DIM), dtype=np.float32 )
        heatmaps.append(img_annotated.flatten())
    heatmaps = torch.tensor( np.array(heatmaps), dtype=torch.float32 )
    return images, heatmaps
def collate_fn_1d_coords(data):
    images, coords, _ = zip(*data)
    coords = np.array(coords)
    images = np.array(images)[:,None,:,:]
    #
    images = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    return images, coords

# Models
## Model01
class Model01(nn.Module):
    def __init__(
            self,
            input_image_shape = (400, 400),
            output_shape=100**2
        ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.input_nx, self.input_ny = self.input_image_shape
        self.output_shape = output_shape
        # Convolutions
        self.conv1_nchannels = 1
        self.conv1_nconvs = 4
        self.conv1_convsize = 25
        self.conv1 = nn.Conv2d(
            self.conv1_nchannels,
            self.conv1_nconvs,
            self.conv1_convsize
        )
        self.conv1_activation = nn.ReLU()
        self.conv1_npooling = 5
        self.conv1_poolingstride = 2
        self.conv1_pooling = nn.MaxPool2d(
            self.conv1_npooling,
            stride=self.conv1_poolingstride
        )
        # TODO: add max_pooling layers
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        # Flatten
        self.flatten = nn.Flatten()
        # Denses
        self.linear1 = nn.Linear(
            in_features=138384,  # TODO calculate this based on parameters above, e.g. self.conv1_convsize, ...
            out_features=512
        )
        self.linear1_activation = nn.ReLU()
        self.dense = nn.Linear(
            in_features=512,
            out_features=self.output_shape
        )
        self.to_probability = nn.Sigmoid()
        return

    def forward(self, x):
        # Convolutions
        x = self.conv1_activation(self.conv1(x))
        x = self.conv1_pooling(x)
        # Flattern
        x = self.flatten(x)
        # Dense
        x = self.linear1_activation(
            self.linear1(x)
        )
        x = self.dense(x)
        return self.to_probability(x)
## Model02
class Model02(nn.Module):
    def __init__(
            self,
            input_image_shape = (400, 400),
        ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.input_nx, self.input_ny = self.input_image_shape
        # Convolutions
        self.conv1_nchannels = 1
        self.conv1_nconvs = 4
        self.conv1_convsize = 25
        self.conv1 = nn.Conv2d(
            self.conv1_nchannels,
            self.conv1_nconvs,
            self.conv1_convsize
        )
        self.conv1_activation = nn.ReLU()
        self.conv1_npooling = 5
        self.conv1_poolingstride = 2
        self.conv1_pooling = nn.MaxPool2d(
            self.conv1_npooling,
            stride=self.conv1_poolingstride
        )
        # TODO: add max_pooling layers
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        # Flatten
        self.flatten = nn.Flatten()
        # Denses
        self.linear1 = nn.Linear(
            in_features=138384,  # TODO calculate this based on parameters above, e.g. self.conv1_convsize, ...
            out_features=512
        )
        self.linear1_activation = nn.ReLU()
        self.dense = nn.Linear(
            in_features=512,
            out_features=2
        )
        return

    def forward(self, x):
        # Convolutions
        x = self.conv1_activation(self.conv1(x))
        x = self.conv1_pooling(x)
        # Flattern
        x = self.flatten(x)
        # Dense
        x = self.linear1_activation(
            self.linear1(x)
        )
        x = self.dense(x)
        return x

## Trainers
