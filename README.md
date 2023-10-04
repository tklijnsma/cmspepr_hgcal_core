# cmspepr_hgcal_core

Data pipelines, models, and loss functions for the HGCAL ML4RECO project.


## Installation and requirements.

The package minimally requires `torch` and `torch_geometric`.
For running the GravNet model, the [torch_cmspepr](https://github.com/cms-pepr/pytorch_cmspepr) package is required. 

The package is not (yet) availabe on PyPI, so installation should be done manually:

```bash
git clone git@github.com:tklijnsma/cmspepr_hgcal_core.git
cd cmspepr_hgcal_core
pip install -e .
```


## Example

```py
import os

import torch
import tqdm
from torch_geometric.loader import DataLoader

import cmspepr_hgcal_core
from cmspepr_hgcal_core.datasets import Taus2021Dataset
from cmspepr_hgcal_core.gravnet_model import GravnetModelWithNoiseFilter
from cmspepr_hgcal_core.objectcondensation import oc_loss_with_noise_filter
from cmspepr_hgcal_core.utils import assert_no_nans, LossResult

logger = cmspepr_hgcal_core.logger

device = torch.device('cpu') # or device = torch.device('cuda')

batch_size = 4
test_loader = DataLoader(Taus2021Dataset('/taus/test'), batch_size=batch_size)
train_loader = DataLoader(Taus2021Dataset('/taus/train'), batch_size=batch_size)

# Throw all but 10 events away for both train and test to showcase functionality.
# Do not do this for an actual training.
test_loader.dataset.npzs = test_loader.dataset.npzs[:10]
train_loader.dataset.npzs = train_loader.dataset.npzs[:10]

model = GravnetModelWithNoiseFilter(9, output_dim=3, k=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

def train():
    for data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out_noise_filter, pass_noise_filter, out_gravnet = model(data)
        assert_no_nans(out_noise_filter)
        assert_no_nans(out_gravnet)
        loss_result = oc_loss_with_noise_filter(out_gravnet, data, pass_noise_filter)
        loss_result.backward()
        optimizer.step()

best_test_loss = torch.inf

def test():
    global best_test_loss
    with torch.no_grad():
        model.eval()
        loss = LossResult()
        for data in tqdm.tqdm(test_loader):
            out_noise_filter, pass_noise_filter, out_gravnet = model(data)
            loss += oc_loss_with_noise_filter(out_gravnet, data, pass_noise_filter)
        loss /= len(test_loader)
        logger.info(f"test loss:\n{loss}")

    if loss.loss < best_test_loss:
        ckpt = "best.pth.tar"
        logger.info(f"Saving to {ckpt}")
        best_test_loss = loss.loss
        os.makedirs("models", exist_ok=True)
        torch.save(dict(model=model.state_dict()), ckpt)

def test():
    for data in tqdm.tqdm(test_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out_noise_filter, pass_noise_filter, out_gravnet = model(data)
        assert_no_nans(out_noise_filter)
        assert_no_nans(out_gravnet)
        loss_result = oc_loss_with_noise_filter(out_gravnet, data, pass_noise_filter)
        loss_result.backward()
        optimizer.step()

for i_epoch in range(10):
    logger.info(f'Epoch {i_epoch}')
    train()
    test()
```

## Tests

```bash
pytest tests/
```
