# Modules
import os
import gc

import numpy as np

from tqdm import tqdm
import datetime
from h5py import File as h5File

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv

import matplotlib.pyplot as plt

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
            gamma_max_deviation = 0.05,
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

        self.gamma_max_deviation = gamma_max_deviation
        self.always_same_rows = always_same_rows
        self.verbose = verbose
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
        if y_idx >= ny or x_idx >= nx or y_idx < 0 or x_idx < 0:
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
            # No Worm
            if label == 0:
                coords_new = np.array(
                    [self.crop_size//2, self.crop_size//2],
                    dtype=np.float32
                )
            if self.verbose:
                print(f"Image IDX: {idx} Label: {label}, COORD NEW Cropped: {coords_new}")
            return img_processed, coords_new, label
        print("DEBUG: Attempts to Augment failed!")
        print(f"Image IDX: {idx} Label: {label}, Coords: {coords_new}")
        raise NotImplementedError()
# DataLoaders
class AnnotatedDataLoaderDebug(Dataset):
    # Constructor
    def __init__(
            self,
            fp_annotated_data,
            factor_augmentations = 1,
            windowsize_min_included = 120,
            crop_size = 400,
            gamma_max_deviation = 0.05,
            always_same_rows = False,
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

        self.gamma_max_deviation = gamma_max_deviation
        self.always_same_rows = always_same_rows
        self.verbose = verbose
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
        if y_idx >= ny or x_idx >= nx or y_idx < 0 or x_idx < 0:
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

        # Info
        info = np.array([ idx, x_idx, y_idx, label, theta, gamma ], dtype=np.float32)

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
            # No Worm
            if label == 0:
                coords_new = np.array(
                    [self.crop_size//2, self.crop_size//2],
                    dtype=np.float32
                )
            if self.verbose:
                print(f"Image IDX: {idx} Label: {label}, COORD NEW Cropped: {coords_new}")
            return img_processed, coords_new, label, info
        print("DEBUG: Attempts to Augment failed!")
        print(f"Image IDX: {idx} Label: {label}, Coords: {coords_new}")
        raise NotImplementedError()

# Collator Functions
def collate_fn_1d_coords(data):
    images, coords, _ = zip(*data)
    coords = np.array(coords)
    images = np.array(images)[:,None,:,:]
    images = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    return images, coords
def collate_fn_1d_coords_normalized(data):
    images, coords, _ = zip(*data)
    coords = np.array(coords) / 400.0
    images = np.array(images)[:,None,:,:] / 255.0
    images = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    return images, coords
def collate_fn_3d_input(data):
    images, coords, _  = zip(*data)
    coords = np.array(coords)
    images = np.repeat(
        np.array(images)[:,None,:,:], 3, axis=1
    )
    images_channeled = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    return images_channeled, coords
def collate_fn_3d_input_debug(data):
    images, coords, _, infos = zip(*data)
    coords = np.array(coords)
    images = np.repeat(
        np.array(images)[:,None,:,:], 3, axis=1
    )
    images_channeled = torch.tensor( images, dtype=torch.float32 )
    coords = torch.tensor( coords, dtype=torch.float32 )
    infos = torch.tensor( np.array(infos), dtype=torch.float32 )
    return images_channeled, coords, infos
def collate_fn_heatmap_generator(output_image_dim = 100, weighted = True, blur_size = 11, mask_size = 16):
    def collate_fn_heatmap(data):
        _n = output_image_dim * output_image_dim
        images, coords, labels = zip(*data)
        _img = images[0]
        images = np.array(images)[:,None,:,:]
        images = torch.tensor( images, dtype=torch.float32 )
        heatmaps, weights = [], []
        weights_constant = np.ones( _n, dtype=np.float32 )
        for (i,j), label in zip(coords,labels):
            if label != 0:
                img_annotated = np.zeros_like( _img, dtype=np.float32 )
                img_annotated[j,i] = 100.0
                #############################################################################
                #### WEIGHTS ####
                if weighted:
                    mask = cv.GaussianBlur(img_annotated,(mask_size,mask_size), 0) > 0.0
                    weights.append( np.array(mask, dtype=np.float32).flatten() )
                else:
                    weights.append( weights_constant )
                #############################################################################
                img_annotated = cv.GaussianBlur(img_annotated,(blur_size,blur_size), 0)
                img_annotated = cv.resize(img_annotated, (output_image_dim, output_image_dim))
                img_annotated /= img_annotated.max()
                
                heatmaps.append(img_annotated.flatten())
            else:
                heatmaps.append(np.zeros( _n, dtype=np.float32 ))
                #############################################################################
                #### WEIGHTS ####
                _weights = np.zeros(_n, dtype=np.float32)
                _indices = np.random.randint( 0, _n, mask_size*mask_size )
                _weights[_indices] = 1.0
                weights.append( _weights )
                #############################################################################
        heatmaps = torch.tensor( np.array(heatmaps), dtype=torch.float32 )
        return images, heatmaps
    return collate_fn_heatmap


# Models


# Trainers
## Coordinates Trainer
class TrainerCoordinates:
    # Constructor
    def __init__(
            self,
            model,
            data_loader_train,
            data_loader_validation = None,
            optimizer_name = 'adam',
            optimizer_lr = 1e-3,
            device = 'auto',
            fp_checkpoints = None
        ) -> None:
        self.model = model
        self.data_loader_train = data_loader_train
        self.data_loader_validation = data_loader_validation if data_loader_validation is not None else self.data_loader_train
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer_name = optimizer_name.lower()
        self.optimizer_lr = optimizer_lr
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_lr)
        else:
            raise NotImplementedError()
        self.device = device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        # Base Path
        assert fp_checkpoints is not None, 'Checkpoints path should be provided.'
        _date_key = str(datetime.datetime.now())[:19].replace(':','').replace(' ','_')
        self.fp_checkpoints = os.path.join(
            fp_checkpoints,
            _date_key
        )
        return
    # Train
    def train(self, n_epochs, n_epochs_checkpoint = 10):
        model = self.model.to(self.device)
        loss_fn = self.loss_fn
        logs = []
        for i_epoch in range(n_epochs):
            print(f"### Epoch: {i_epoch+1:>3}/{n_epochs:<3}")
            losses_epoch_train = []

            # Train
            steps = tqdm(self.data_loader_train, desc=f'Epoch Steps - Train Loss: {0.0:>7.3f}')
            model.train()
            for x_train, y_train in steps:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                y_train_pred = model(x_train)

                # Compute the loss and its gradients
                loss = loss_fn(y_train_pred, y_train)
                loss.backward()
                loss_value = loss.cpu().item()

                # Adjust learning weights
                self.optimizer.step()

                # Log
                losses_epoch_train.append(loss_value)
                steps.set_description(
                    'Epoch Steps - Train Loss: {:>7.3f}'.format(loss_value)
                )
            # Validation
            losses_epoch_validation = []
            model.eval()
            with torch.no_grad():
                steps = tqdm(self.data_loader_validation, desc=f'Epoch Steps - Validation Loss: {0.0:>7.3f}')
                for x_train, y_train in steps:
                    x_train = x_train.to(self.device)
                    y_train = y_train.to(self.device)
                    # Make predictions for this batch
                    y_train_pred = model(x_train)

                    # Compute the loss and its gradients
                    loss = loss_fn(y_train_pred, y_train)
                    loss_value = loss.cpu().item()

                    # Log
                    losses_epoch_validation.append(loss_value)
                    steps.set_description(
                        'Epoch Steps - Validation Loss: {:>7.3f}'.format(loss_value)
                    )
            logs.append([
                np.mean(losses_epoch_train), np.mean(losses_epoch_validation),
                losses_epoch_train, losses_epoch_validation
            ])
            # Save Model
            if (i_epoch+1)%n_epochs_checkpoint == 0:
                if not os.path.exists(self.fp_checkpoints):
                    os.mkdir(self.fp_checkpoints)
                fp_model = os.path.join( self.fp_checkpoints, str(i_epoch).zfill(3)+".pt" )
                torch.save({
                    'epoch': i_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_train': np.mean(losses_epoch_train),
                    'loss_validation': np.mean(losses_epoch_validation),
                    },
                    fp_model
                )
            # Garbage Collect
            _ = gc.collect()
        return logs
    # Test
    def test(self, data_loader_test):
        model = self.model.to(self.device)
        loss_fn = self.loss_fn

        # Test
        losses_epoch_test = []
        model.eval()
        with torch.no_grad():
            steps = tqdm(data_loader_test, desc=f'Epoch Steps - Test Loss: {0.0:>7.3f}')
            for x_train, y_train in steps:
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)
                # Make predictions for this batch
                y_train_pred = model(x_train)

                # Compute the loss and its gradients
                loss = loss_fn(y_train_pred, y_train)
                loss_value = loss.cpu().item()

                # Log
                losses_epoch_test.append(loss_value)
                steps.set_description(
                    'Epoch Steps - Test Loss: {:>7.3f}'.format(loss_value)
                )
            # Garbage Collect
            _ = gc.collect()
        return losses_epoch_test