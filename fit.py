#!/usr/bin/env python
# coding: utf-8


import numpy as np
import json
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import load_config


class Model(nn.Module):
    def __init__(self, in_size, hidden_channel, hidden_size, kernel_1, hidden_channel_2, kernel_2, hidden_channel_3,
                 kernel_3):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(in_size, hidden_channel * hidden_size * hidden_size)
        self.c1 = nn.ConvTranspose2d(hidden_channel, hidden_channel_2, kernel_1)
        self.c2 = nn.ConvTranspose2d(hidden_channel_2, hidden_channel_3, kernel_2)
        self.c3 = nn.ConvTranspose2d(hidden_channel_3, 1, kernel_3)

    def forward(self, x):
        x = self.l1(x).reshape((x.shape[0], self.hidden_channel, self.hidden_size, self.hidden_size))
        x = F.relu(x)
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = x.squeeze()
        return x


class ModelCommittee(nn.Module):
    def __init__(self, count, **kwargs):
        super().__init__()
        models = []
        for i in range(count):
            models.append(Model(**kwargs))
        self.models = nn.ModuleList(models)

    def load_dict(self, template):
        for i, m in enumerate(self.models):
            m.load_state_dict(torch.load(template.format(drop=i)))

    def forward(self, x):
        res = []
        for m in self.models:
            res.append(m(x))

        return torch.mean(torch.stack(res, axis=-1), axis=-1)


def fit():
    # Load Config
    config = load_config()['Fit Model']

    # Load Training Data
    data = np.load('data.npy', allow_pickle=True)
    data, labels = data.item()['configs'], data.item()['labels']

    # Setup cuda device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Convert to Torch tensor
    X = torch.from_numpy(data).to(device=device, dtype=torch.float)
    Y = torch.from_numpy(labels).to(device=device, dtype=torch.float)
    Y = Y[:, slice(*config['Subsample Slice']['x']), slice(*config['Subsample Slice']['y'])]

    # Compute normalization data
    Xm = X.mean()
    Xstd = X.std()
    X = (X - Xm) / Xstd
    Ym = Y.mean()
    Ystd = Y.std()
    Y = (Y - Ym) / Ystd
    normalizer = {"Xm": Xm.detach().cpu(), "Xstd": Xstd.detach().cpu(), "Ym": Ym.detach().cpu(),
                  "Ystd": Ystd.detach().cpu()}
    torch.save(normalizer, "norm.pt")

    # Training Loop. N fold cross validation
    valLoss = []
    fold = config['Training']['Folds']
    size = X.shape[0] // fold
    for drop in range(fold):
        # Sample masks
        mask = torch.zeros(X.shape[0])
        mask[drop * size:(drop + 1) * size] = 1
        mask = mask.to(dtype=bool)
        Xp = X[~mask]
        Yp = Y[~mask]

        # Setup model and optimizers
        model = Model(**config['Model']).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), config['Training']['Learning Rate'])
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        mse = torch.nn.MSELoss()

        # Training Loop
        pbar = tqdm(range(config['Training']['Training Epochs']))
        for _ in pbar:
            pred = model(Xp)
            # Compute and print loss
            loss = mse(pred, Yp)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Loss: {loss}')

        # Validate Model
        pred = model(X[mask])
        loss = mse(pred, Y[mask].squeeze())
        valLoss.append(loss)
        print(f"Fold {drop}: {loss}")
        torch.save(model.state_dict(), f"models/Model-CrossVal-{drop}.pt")
        del model
        del optimizer
        del scheduler

    # Save validation data
    np.save('val_loss.npy', np.array(valLoss))

    # Create committee
    com = ModelCommittee(X.shape[1], count=config['Training']['Folds'], **config['Model']).to(device=device)
    com.load_dict("models/Model-CrossVal-{drop}.pt")
    pred = com(X)
    print(f"Committee Loss: {mse(pred, Y)}")


if __name__ == '__main__':
    fit()
