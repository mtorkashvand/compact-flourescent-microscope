# Modules
import datetime
import os

import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from nwtools import AnnotatedDataLoader, collate_fn_1d_coords
from nwtools import Model02 as Model

# Parameters
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

# Methods

# Classes

# Main
dataset = AnnotatedDataLoader(
    fp_annotated_data=r"V:\Mahdi\OpenAutoScope2.0\data\training_data\training_data.h5",
    factor_augmentations = 1
)
dataloader = DataLoader(
    dataset,
    batch_size=32, shuffle=True,
    collate_fn=collate_fn_1d_coords
)

# Training
_date_key = str(datetime.datetime.now())[:19].replace(':','').replace(' ','_')
fp_base = os.path.join(
    r"V:\Mahdi\OpenAutoScope2.0\data\trained_models",
    _date_key
)

model = Model().to(DEVICE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
logs = []
epochs = tqdm( range(10), desc=f'Loss: {0.0:>7.3f}', position=0 )
for i_epoch in epochs:
    losses_epoch = []
    steps = tqdm(dataloader, desc=f'Epoch Steps - Loss: {0.0:>7.3f}',position=1, leave=False)
    for x_train, y_train in steps:
        x_train = x_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_train_pred = model(x_train)

        # Compute the loss and its gradients
        loss = loss_fn(y_train_pred, y_train)
        loss.backward()
        loss_value = loss.cpu().item()

        # Adjust learning weights
        optimizer.step()
        
        # Log
        losses_epoch.append(loss_value)
        steps.set_description(
            'Epoch Steps - Loss: {:>7.3f}'.format(loss_value)
        )
    logs.append([
        np.mean(losses_epoch),
        losses_epoch.copy()
    ])
    # Report
    epochs.set_description(
        'Loss: {:>7.3f}'.format( logs[-1][0] )
    )
    print("\n\n")
    # Save Model
    if not os.path.exists(fp_base):
        os.mkdir(fp_base)
    fp_model = os.path.join( fp_base, str(i_epoch).zfill(3)+".pt" )
    torch.save({
        'epoch': i_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.mean(losses_epoch),
        },
        fp_model
    )
